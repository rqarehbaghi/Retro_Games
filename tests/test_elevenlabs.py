import base64
import os
import tempfile
import unittest
from unittest.mock import patch
from tools.progression import speech


class ElevenTests(unittest.TestCase):
    def test_voice_ids_not_files(self):
        speech.validate_voice('elevenlabs','HostOne,HostTwo',True)
        for refs in ('host1.wav,host2.wav','Same,Same','OnlyOne'):
            with self.assertRaises(ValueError):
                speech.validate_voice('elevenlabs',refs,True)

    def test_dialogue_bounds_preserve_gaps_and_refuse_overlap(self):
        segments=[dict(dialogue_input_index=0,start_time_seconds=.1,end_time_seconds=1),
                  dict(dialogue_input_index=1,start_time_seconds=1.2,end_time_seconds=2)]
        self.assertEqual(speech._eleven_bounds(segments,2,2.1),[(0,1.2),(1.2,2.1)])
        segments[1]['start_time_seconds']=.5
        with self.assertRaisesRegex(RuntimeError,'Overlapping'):
            speech._eleven_bounds(segments,2,2.1)
        with self.assertRaisesRegex(RuntimeError,'omitted'):
            speech._eleven_bounds([],1,2)

    def test_free_guard_and_quota_before_generation(self):
        for sub in (dict(tier='starter'),dict(tier='free',character_limit=1,character_count=1)):
            with tempfile.TemporaryDirectory() as out, patch.object(speech,'_eleven_request',return_value=sub) as request:
                with self.assertRaises(RuntimeError):
                    speech._speak_elevenlabs(['Hello'],out,'HostOne')
                request.assert_called_once_with('/v1/user/subscription')

    def test_joint_generation_and_cached_retry(self):
        response=dict(audio_base64=base64.b64encode(b'audio').decode(),voice_segments=[
            dict(dialogue_input_index=0,start_time_seconds=0,end_time_seconds=1),
            dict(dialogue_input_index=1,start_time_seconds=1,end_time_seconds=2)])
        with tempfile.TemporaryDirectory() as out, \
             patch.object(speech,'_eleven_request',side_effect=[dict(tier='free',character_limit=10000,character_count=0),response]) as request, \
             patch.object(speech,'seconds',return_value=2),patch.object(speech.subprocess,'run'):
            paths=speech._speak_elevenlabs(['Hello','Hi'],out,['HostOne','HostTwo'])
            self.assertEqual(len(paths),2)
            self.assertEqual(request.call_count,2)
            payload=request.call_args.args[1]
            self.assertEqual([t['voice_id'] for t in payload['inputs']],['HostOne','HostTwo'])
            speech._speak_elevenlabs(['Hello','Hi'],out,['HostOne','HostTwo'])
            self.assertEqual(request.call_count,2)
            with self.assertRaises(ValueError):
                speech._speak_elevenlabs(['x'*2001],out,'HostOne')

    def test_edit_plan_routes_two_hosts_and_preserves_silent_movie(self):
        from pathlib import Path
        from tools.progression import story
        panels=[dict(key=k+'_a_bare',state='a',value=10,seconds=11,stats=[['LINES',10]],
                     path=k+'.mp4',end_reason='game_over',timeline=[dict(t=10,v=10,d=10,end='game_over')])
                for k in ('10k','20k')]
        def film(who):
            return dict(panels=[p['key'] for p in panels],scenes=[dict(start=0,duration=11)],
                        narration=[dict(who=who,text='Hello there.')])
        plan=dict(films=dict(long=film(2),short=film(1)))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(story.speech,'available',return_value=True), \
             patch.object(story.speech,'speak',return_value=[('turn.wav',1)]) as speak, \
             patch.object(story,'render_scene'),patch.object(story,'concat'), \
             patch.object(story.video,'duration',return_value=11), \
             patch.object(story.narrate,'build_track'),patch.object(story.narrate,'mux') as mux:
            story.render_edit_plan(plan,panels,Path(tmp)/'out','elevenlabs',['HostOne','HostTwo'])
            self.assertEqual([c.kwargs['voice'] for c in speak.call_args_list],[['HostTwo'],['HostOne']])
            self.assertEqual(mux.call_count,2)


if __name__=='__main__':
    unittest.main()
