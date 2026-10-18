import struct
import pytest
from panoprivacy.mp4_metadata import pack, boxes, read_moov, spherical_boxes, restore_spherical_metadata

UUID = bytes.fromhex('ffcc8263f8554a938814587a02521fdd')
def movie(sphere=True):
    sv = pack(b'sv3d', pack(b'svhd',bytes(4)+b'PanoramaPrivacyTest\0') +
              pack(b'proj',pack(b'prhd',bytes(16))+pack(b'equi',bytes(20)))) if sphere else b''
    entry = pack(b'hvc1', bytes(78) + sv)
    stsd = pack(b'stsd', bytes(4) + struct.pack('>I', 1) + entry)
    track = pack(b'trak', pack(b'mdia', pack(b'minf', pack(b'stbl', stsd))) + (pack(b'uuid', UUID + b'<ProjectionType>equirectangular</ProjectionType>') if sphere else b''))
    return pack(b'moov', track)

def test_metadata_restore_preserves_media_offsets_and_source(tmp_path):
    source, encoded, output = [tmp_path / n for n in ('source.mp4','encoded.mp4','result.mp4')]
    source.write_bytes(pack(b'ftyp',b'isom0000') + movie())
    original = pack(b'ftyp',b'isom0000') + movie(False) + pack(b'mdat',b'unchanged payload')
    encoded.write_bytes(original)
    restore_spherical_metadata(source, encoded, output)
    assert encoded.read_bytes() == original
    assert output.read_bytes().index(b'unchanged payload') == original.index(b'unchanged payload')
    assert spherical_boxes(output) == spherical_boxes(source)
    assert read_moov(output)[0] == len(original)
    with pytest.raises(FileExistsError): restore_spherical_metadata(source, encoded, output)

def test_malformed_boxes_rejected():
    with pytest.raises(ValueError): list(boxes(b'\x00\x00\x00\x20moov'))

def test_large_mdat_is_skipped_with_64bit_offsets_without_reading_payload(monkeypatch):
    """A virtual >4GiB file asserts that parsing only reads headers and moov."""
    from pathlib import Path
    from types import SimpleNamespace
    moov=movie()
    mdat_size=2**32+4096
    header=struct.pack('>I4sQ',1,b'mdat',mdat_size)
    class VirtualFile:
        position=0
        maximum_read=0
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def tell(self): return self.position
        def seek(self,position): self.position=position
        def read(self,count):
            self.maximum_read=max(self.maximum_read,count)
            if self.position<16: data=header[self.position:self.position+count]
            elif self.position>=mdat_size: data=moov[self.position-mdat_size:self.position-mdat_size+count]
            else: raise AssertionError('Parser tried to read giant mdat payload')
            self.position+=len(data); return data
    virtual=VirtualFile()
    monkeypatch.setattr(Path,'stat',lambda self:SimpleNamespace(st_size=mdat_size+len(moov)))
    monkeypatch.setattr(Path,'open',lambda self,*args,**kwargs:virtual)
    offset,size,data=read_moov(Path('virtual.mp4'))
    assert offset==mdat_size and data==moov and size==len(moov)
    assert virtual.maximum_read<=len(moov)
