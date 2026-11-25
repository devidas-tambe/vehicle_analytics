param(
    [Parameter(Mandatory=$true)][string]$Video,
    [string]$Url="rtsp://127.0.0.1:8554/traffic"
)
ffmpeg -re -stream_loop -1 -i $Video -an -c:v libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -g 30 -f rtsp -rtsp_transport tcp $Url
