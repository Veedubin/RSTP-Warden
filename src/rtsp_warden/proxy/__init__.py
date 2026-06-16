"""Proxy backends.

Currently supported:
* mjpeg (HTTP) - implemented via ffmpeg -> stdout parser -> ThreadingHTTPServer
* rtsp (MediaMTX) - one MediaMTX instance per camera/port
"""
