"""Encoder dispatch tests use a mocked hardware probe and never acquire a GPU."""
import threading
import pytest
from panoprivacy.common import Cancelled
from panoprivacy import video_engine as engine

def asset():
    return {'width':7680,'height':3840,'pix_fmt':'yuv420p10le','depth':10,'fps_num':30,'fps_den':1}

def test_auto_encoder_uses_actual_probe_and_records_selection(monkeypatch):
    data=asset(); events=[]
    monkeypatch.setattr(engine,'_probe_nvenc',lambda a,c:None)
    selected=engine.select_video_encoder(data,events.append)
    assert selected=='hevc_nvenc'
    assert data['encoder_probe']['encoder']=='hevc_nvenc'
    assert data['encoder_probe']['width']==7680
    assert 'NVENC' in events[-1]['message']

def test_auto_encoder_reports_cpu_fallback_on_probe_failure(monkeypatch):
    data=asset(); events=[]
    def unavailable(a,c): raise RuntimeError('No NVENC capable device found')
    monkeypatch.setattr(engine,'_probe_nvenc',unavailable)
    assert engine.select_video_encoder(data,events.append)=='libx265'
    assert data['encoder_probe']['fallback'] is True
    assert 'No NVENC' in data['encoder_probe']['reason']
    assert 'CPU' in events[-1]['message']

def test_auto_encoder_does_not_turn_cancellation_into_cpu_fallback(monkeypatch):
    data=asset()
    def cancelled(a,c): raise Cancelled('cancelled')
    monkeypatch.setattr(engine,'_probe_nvenc',cancelled)
    with pytest.raises(Cancelled): engine.select_video_encoder(data)
    assert 'encoder_probe' not in data

def test_explicit_cpu_override_does_not_probe_gpu(monkeypatch):
    data=dict(asset(),export_encoder='libx265')
    def forbidden(*args): raise AssertionError('explicit selection probed hardware')
    monkeypatch.setattr(engine,'_probe_nvenc',forbidden)
    assert engine.select_video_encoder(data)=='libx265'

def test_probe_command_uses_actual_spec_single_frame_and_timeout(monkeypatch):
    calls=[]
    monkeypatch.setattr(engine,'run_checked',lambda command,**kwargs:calls.append((command,kwargs)))
    engine._probe_nvenc(asset(),None)
    command,kwargs=calls[0]
    assert 'color=c=black:s=7680x3840:r=30/1' in command
    assert command[command.index('-frames:v')+1]=='1'
    assert command[command.index('-pix_fmt')+1]=='yuv420p10le'
    assert kwargs['timeout']==15

def test_process_timeout_kills_child_promptly():
    import sys,time
    from panoprivacy.video_process import run_checked
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        run_checked([sys.executable,'-c','import time; time.sleep(60)'],timeout=.1)
    assert time.monotonic()-started<3

def test_process_cancellation_kills_child_and_retains_cancelled_exception():
    import sys,time
    from panoprivacy.video_process import run_checked
    cancel=threading.Event()
    timer=threading.Timer(.1,cancel.set)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            run_checked([sys.executable,'-c','import time; time.sleep(60)'],cancel=cancel,timeout=2)
    finally:
        timer.cancel()
