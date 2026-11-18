import asyncio, io, json, os, unittest
from unittest.mock import patch
import httpx
import numpy as np
import soundfile as sf
from api.backends.cuda_graphs import CUDAGraphBackend

class SplitStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        raw=np.array([.1,.2,.3],dtype='<f4').tobytes()
        for block in [raw[:3],raw[3:7],raw[7:]]:yield block

class GraphTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Preserve the ambient loop used by this repository's legacy sync tests.
        try: cls._previous_loop=asyncio.get_event_loop()
        except RuntimeError: cls._previous_loop=None

    @classmethod
    def tearDownClass(cls):
        asyncio.set_event_loop(cls._previous_loop)

    def client_patch(self, handler):
        cls=httpx.AsyncClient
        return patch('api.backends.cuda_graphs.httpx.AsyncClient',side_effect=lambda **kw:cls(transport=httpx.MockTransport(handler)))

    async def test_stream_handles_arbitrary_network_boundaries(self):
        def handler(req):return httpx.Response(200,stream=SplitStream())
        with self.client_patch(handler):
            b=CUDAGraphBackend('checkpoint');parts=[a async for a,sr in b.generate_speech_streaming('Hi','Ryan')]
        np.testing.assert_allclose(np.concatenate(parts),[.1,.2,.3])

    async def test_stream_rejects_truncated_sample(self):
        with self.client_patch(lambda req:httpx.Response(200,content=b'x')):
            with self.assertRaisesRegex(RuntimeError,'truncated'):
                async for _ in CUDAGraphBackend('m').generate_speech_streaming('Hi','Ryan'):pass

    async def test_worker_error_propagates(self):
        with self.client_patch(lambda req:httpx.Response(503,content=b'offline')):
            with self.assertRaises(httpx.HTTPStatusError):
                await CUDAGraphBackend('m').generate_speech('Hi','Ryan')

    async def test_wav_and_instructions_roundtrip(self):
        buf=io.BytesIO();sf.write(buf,np.ones(240,dtype=np.float32)*.1,24000,format='WAV');seen=[]
        def handler(req):seen.append(json.loads(req.content));return httpx.Response(200,content=buf.getvalue())
        with self.client_patch(handler):a,sr=await CUDAGraphBackend('m').generate_speech('Hi','Ryan',instruct='Whisper')
        self.assertEqual(sr,24000);self.assertEqual(len(a),240);self.assertEqual(seen[0]['instruct'],'Whisper')

    async def test_health_discovers_native_voices(self):
        h={'ready':True,'voices':['ryan','serena'],'languages':['english'],'device':{'gpu_name':'RTX 3060','type':'cuda:0'}}
        with self.client_patch(lambda req:httpx.Response(200,json=h)):
            b=CUDAGraphBackend('m');await b.initialize()
        self.assertTrue(b.is_ready());self.assertEqual(b.get_supported_voices(),h['voices']);self.assertEqual(b.get_device_info()['gpu_name'],'RTX 3060');self.assertEqual(b.get_device_info()['device'],'cuda:0')

if __name__=='__main__':unittest.main()
