"""DeepStream 7.1 / PyDS 1.2.0 single-RTSP vehicle analytics pipeline."""
import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.counting import LineCounter
from vision.prepare import prepare
from vision.writer import EventWriter

WIDTH, HEIGHT = 1280, 720
UNTRACKED = 0xFFFFFFFFFFFFFFFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="rtsp://127.0.0.1:8554/traffic")
    parser.add_argument("--source-id", default="camera-1")
    parser.add_argument("--line-y", type=float, default=0.55)
    parser.add_argument(
        "--publish",
        help="Optional annotated RTSP output URI, requires encoder and rtspclientsink",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    counter = LineCounter(args.line_y)
    pgie_cfg, sgie_cfg, tracker_cfg, tracker_lib = prepare()

    os.environ.pop("USE_NEW_NVSTREAMMUX", None)
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds

    Gst.init(None)
    run_id = uuid.uuid4()
    vehicle_classes = {int(i) for i in os.getenv("VEHICLE_CLASS_IDS", "0").split(";")}
    loop = GLib.MainLoop()
    pipe = Gst.Pipeline.new("vehicle-analytics")
    fatal = []

    def make(factory, name, **props):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(
                f"Missing GStreamer element {factory}; check DeepStream installation"
            )
        for key, value in props.items():
            element.set_property(key.replace("_", "-"), value)
        pipe.add(element)
        return element

    def link(*elements):
        for a, b in zip(elements, elements[1:]):
            if not a.link(b):
                raise RuntimeError(f"Could not link {a.name} to {b.name}")

    source = make("uridecodebin", "input", uri=args.uri)
    mux = make(
        "nvstreammux",
        "mux",
        width=WIDTH,
        height=HEIGHT,
        batch_size=1,
        live_source=1,
        batched_push_timeout=40000,
        enable_padding=False,
    )
    sinkpad = (
        mux.request_pad_simple("sink_0")
        if hasattr(mux, "request_pad_simple")
        else mux.get_request_pad("sink_0")
    )
    if sinkpad is None:
        raise RuntimeError("nvstreammux request pad failed")

    def child_added(parent, child, name):
        if "decodebin" in name and child.find_property("caps") is not None:
            child.connect("child-added", child_added)
        if child.find_property("protocols") is not None and "source" in name:
            child.set_property("protocols", 4)  # RTSP over TCP

    source.connect("child-added", child_added)

    def fail(message):
        logging.error(message)
        fatal.append(message)
        loop.quit()
        return False

    def pad_added(_, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        if not caps.get_features(0).contains("memory:NVMM"):
            GLib.idle_add(
                fail, "Decoder did not expose NVMM surfaces; NVIDIA decoder required"
            )
            return
        if not sinkpad.is_linked() and pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            GLib.idle_add(fail, "Could not link decoder to streammux")

    source.connect("pad-added", pad_added)
    pgie = make("nvinfer", "vehicle-detector", config_file_path=str(pgie_cfg))
    tracker = make(
        "nvtracker",
        "tracker",
        tracker_width=640,
        tracker_height=384,
        ll_lib_file=str(tracker_lib),
        ll_config_file=str(tracker_cfg),
    )
    sgie = make("nvinfer", "vehicle-color", config_file_path=str(sgie_cfg))
    conv = make("nvvideoconvert", "rgba-converter")
    caps = make(
        "capsfilter",
        "rgba-caps",
        caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"),
    )
    osd = make("nvdsosd", "overlay", process_mode=0)
    link(mux, pgie, tracker, sgie, conv, caps, osd)

    if args.publish:
        outconv = make("nvvideoconvert", "encoder-converter")
        outcaps = make(
            "capsfilter",
            "encoder-caps",
            caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"),
        )
        encoder = make("nvv4l2h264enc", "h264", bitrate=2500000)
        parse = make("h264parse", "parser", config_interval=-1)
        out = make("rtspclientsink", "publisher", location=args.publish, protocols=4)
        link(osd, outconv, outcaps, encoder, parse, out)
    else:
        sink = make("fakesink", "sink", sync=False)
        sink.set_property("async", False)
        link(osd, sink)

    writer = EventWriter(
        os.getenv(
            "DATABASE_URL", "postgresql://vehicle:vehicle_dev@127.0.0.1:5432/vehicles"
        ),
        mode="live",
    )
    colors = {}

    def walk(head, cast):
        while head is not None:
            try:
                yield cast(head.data)
                head = head.next
            except StopIteration:
                return

    def probe(_, info):
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch is None:
                return Gst.PadProbeReturn.OK
            for frame in walk(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
                source_id = f"{args.source_id}:{frame.source_id}"
                for obj in walk(frame.obj_meta_list, pyds.NvDsObjectMeta.cast):
                    if obj.class_id not in vehicle_classes or obj.object_id == UNTRACKED:
                        obj.rect_params.border_width = 0
                        obj.text_params.display_text = ""
                        continue
                    key = (source_id, int(obj.object_id))
                    for cm in walk(obj.classifier_meta_list, pyds.NvDsClassifierMeta.cast):
                        if cm.unique_component_id != 2:
                            continue
                        for label in walk(cm.label_info_list, pyds.NvDsLabelInfo.cast):
                            color = label.result_label.strip().lower()
                            score = float(label.result_prob)
                            if (
                                color
                                and 0 <= score <= 1
                                and score >= colors.get(key, ("unknown", -1))[1]
                            ):
                                colors[key] = (color, score)
                    color, score = colors.get(key, ("unknown", None))
                    obj.text_params.display_text = f"ID {obj.object_id} | {color}"
                    y = (obj.rect_params.top + obj.rect_params.height) / HEIGHT
                    direction = counter.observe(key, y)
                    if direction:
                        writer.submit(
                            (
                                run_id,
                                source_id,
                                str(obj.object_id),
                                color,
                                score,
                                datetime.now(timezone.utc),
                                direction,
                            )
                        )
                display = pyds.nvds_acquire_display_meta_from_pool(batch)
                display.num_lines = 1
                line = display.line_params[0]
                line.x1 = 0
                line.x2 = WIDTH - 1
                line.y1 = line.y2 = int(HEIGHT * args.line_y)
                line.line_width = 3
                line.line_color.set(1.0, 0.7, 0.0, 1.0)
                pyds.nvds_add_display_meta_to_frame(frame, display)
        except Exception as exc:
            logging.exception("Probe failure")
            GLib.idle_add(fail, str(exc))
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    osd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)

    def message(_, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            fail(f"{err}: {debug}")
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", message)
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(loop.quit))
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(loop.quit))
    logging.info("Run %s; input %s; boundary %.2f", run_id, args.uri, args.line_y)
    try:
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Could not start pipeline")
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
        mux.release_request_pad(sinkpad)
        bus.remove_signal_watch()
        flushed = writer.close()
    return 1 if fatal or not flushed else 0


if __name__ == "__main__":
    raise SystemExit(main())