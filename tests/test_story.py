"""Cached-story contracts; no emulator, account, or speech models needed."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.progression import story
from tools.progression.speech_quality import assess, words


def panel(version, state, value, seconds=100):
    return dict(key=version+'_'+state+'_bare',state=state,value=value,seconds=seconds+1,
                end_reason='game_over',decisions=50,timeline=[{'t':seconds,'end':'game_over'}])


class StoryTests(unittest.TestCase):
    def setUp(self):
        self.run = dict(states=['a','b'],columns=[['one','10k','10k steps'],
                         ['two','20k','20k steps'],['three','30k','30k steps']])
        self.panels = [panel('10k','a',1),panel('10k','b',3),
                       panel('20k','a',9),panel('20k','b',3),
                       panel('30k','a',5),panel('30k','b',5)]

    def test_selects_top_means_but_same_first_state_and_practice_order(self):
        shown,results,selection = story.select(self.panels,self.run)
        self.assertEqual([p['key'] for p in shown],['20k_a_bare','30k_a_bare'])
        self.assertEqual([p['label'] for p in shown],['Less practice','More practice'])
        self.assertEqual(selection['chosen_opponent'],'20k')
        self.assertEqual(len(results),4)

    def test_rejects_missing_pair_and_caps(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            story.select(self.panels[:-1],self.run)
        self.panels[-1]['end_reason'] = 'placement_cap'
        self.panels[-1]['timeline'][-1]['end'] = 'placement_cap'
        with self.assertRaisesRegex(ValueError,'completed'):
            story.select(self.panels,self.run)

    def test_ties_use_manifest_order_and_are_disclosed(self):
        self.panels[-1]['value'] = 7
        _,_,selection = story.select(self.panels,self.run)
        self.assertEqual(selection['tied_best'],['20k','30k'])
        self.assertEqual(selection['chosen_opponent'],'20k')

    def test_empty_suite_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonempty'):
            story.select(self.panels,dict(self.run,states=[]))

    def test_shared_windows_do_not_overlap_and_speed_is_bounded(self):
        for end in (3,40,400,2700):
            panels = [panel('one','a',1,end/2),panel('two','a',2,end)]
            lengths = [7,12,5,10,11,15]
            starts,speed = story.source_windows(panels,lengths)
            self.assertGreaterEqual(speed,1)
            self.assertLessEqual(speed,4)
            self.assertEqual(starts[0],0)
            self.assertTrue(all(b+1e-8 >= a+d*speed
                                for a,b,d in zip(starts,starts[1:],lengths)))
            if end >= sum(lengths):
                self.assertAlmostEqual(starts[-1]+lengths[-1]*speed,end)

    def test_writer_fallback_ending_and_resume_edits(self):
        data = dict(body=[dict(text='A short plain sentence.',who=1)]*3,
                    card=[],closing=dict(text='See you.',who=2))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'prompts').mkdir()
            with patch.object(story.writer,'write',return_value=data):
                result = story.write_script(dict(chosen_opponent='Less practice'),60,
                                            out,'short',{},True)
            self.assertIn('Less practice',result['closing'][0])
            self.assertIn('Stay tuned',result['closing'][0])
            with patch.object(story.writer,'write',side_effect=AssertionError('No rewrite')):
                loaded = story.write_script({},60,out,'short',{},True,True)
            self.assertEqual(loaded['closing'][0],result['closing'][0])

    def test_empty_cleaned_speech_rejected(self):
        data = dict(body=[('okay',1)]*6,card=[('results',1)],
                    closing=('Stay tuned for the next match!',1))
        data['body'][0] = ('',1)
        with self.assertRaisesRegex(ValueError,'1-45'):
            story.validate_script(data)

    def test_speech_count_mismatch_is_fatal_before_render(self):
        data = dict(body=[('Hello',1)]*3,card=[],closing=('Next match',1))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(story.speech,'speak',return_value=[('one.wav',1)]):
            with self.assertRaisesRegex(RuntimeError,'omitted'):
                story.make_film([],{},data,Path(folder),'short',60,['a','b'],'cosyvoice3')

    def test_speech_checks_accept_punctuation_but_reject_inventions(self):
        text = "I'm taking on this computer myself. Stay tuned for the next match!"
        self.assertTrue(assess(text,text.upper().replace("I'M",'I AM'),7)['accepted'])
        self.assertFalse(assess(text,text+' Visit this website for free games and downloads.',12)['accepted'])
        self.assertFalse(assess(text,'',7)['accepted'])
        self.assertFalse(assess(text,text,100)['accepted'])
        self.assertEqual(words('You’re ready.'),['you','are','ready'])

    def test_cosyvoice_retries_mismatch_before_saving(self):
        self._worker_check(['random unrelated extra words', 'Stay tuned for the next match.'],2,True)

    def test_cosyvoice_stops_after_three_mismatches(self):
        self._worker_check(['random unrelated extra words']*3,3,False)

    def _worker_check(self, transcripts, count, success):
        from tools.progression import cosyvoice_runner as worker
        from unittest.mock import Mock
        class Audio:
            ndim = 2
            shape = (1,48000)
            def detach(self): return self
            def cpu(self): return self
        audio=Audio()
        model=SimpleNamespace(sample_rate=24000,
                              inference_instruct2=Mock(return_value=[{'tts_speech':audio}]))
        recognizer=SimpleNamespace(transcribe=Mock(side_effect=transcripts))
        save_audio=Mock()
        modules={'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),
                                        cat=lambda pieces,dim:audio),
                 'torchaudio':SimpleNamespace(save=save_audio),
                 'cosyvoice.cli.cosyvoice':SimpleNamespace(AutoModel=lambda **kw:model),
                 'tools.progression.speech_quality':SimpleNamespace(Recognizer=lambda:recognizer,assess=assess)}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            job=out/'job.json'
            job.write_text(json.dumps(dict(model_dir='model',style='natural',blocks=[
                dict(text='Stay tuned for the next match.',reference=str(out/'ref.wav'),
                     output=str(out/'audio.wav'))])),encoding='utf-8')
            with patch.dict(sys.modules,modules), patch.object(sys,'argv',['worker',str(job)]), \
                    patch.object(sys,'path',list(sys.path)):
                if success:
                    worker.main()
                else:
                    with self.assertRaisesRegex(RuntimeError,'Speech validation failed'):
                        worker.main()
            reports=json.loads((out/'audio.wav.quality.json').read_text(encoding='utf-8'))
            self.assertEqual(len(reports),count)
            self.assertEqual(model.inference_instruct2.call_count,count)
            self.assertEqual(save_audio.call_count,1 if success else 0)

    def test_worker_removes_script_folder_that_shadows_stdlib_select(self):
        from tools.progression import cosyvoice_runner as worker
        folder = str(Path(worker.__file__).resolve().parent)
        with patch.object(sys,'path',[folder]+list(sys.path)):
            worker._prepare_imports()
            self.assertNotIn(folder,sys.path)
            self.assertIn(str(Path(folder).parents[1]),sys.path)

    def test_normalizer_uses_complete_local_cache_and_restores_downloader(self):
        from tools.progression import cosyvoice_runner as worker
        from unittest.mock import Mock
        downloader=Mock(return_value='/cache/wetext')
        module=SimpleNamespace(snapshot_download=downloader,Normalizer=Mock())
        with patch.object(worker.importlib,'import_module',return_value=module), \
                patch.object(worker.os.path,'isfile',return_value=True):
            with worker._cached_normalizer():
                self.assertEqual(module.snapshot_download('pengzhendong/wetext'),'/cache/wetext')
        downloader.assert_called_once_with('pengzhendong/wetext',local_files_only=True)
        self.assertIs(module.snapshot_download,downloader)

    def test_normalizer_incomplete_cache_falls_back_to_download(self):
        from tools.progression import cosyvoice_runner as worker
        from unittest.mock import Mock, call
        downloader=Mock(return_value='/cache/wetext')
        module=SimpleNamespace(snapshot_download=downloader,Normalizer=Mock())
        with patch.object(worker.importlib,'import_module',return_value=module), \
                patch.object(worker.os.path,'isfile',return_value=False):
            with worker._cached_normalizer():
                module.snapshot_download('pengzhendong/wetext')
        self.assertEqual(downloader.call_args_list,[
            call('pengzhendong/wetext',local_files_only=True),call('pengzhendong/wetext')])

    def test_normalizer_uses_legacy_english_assets_without_remote_lookup(self):
        from tools.progression import cosyvoice_runner as worker
        from unittest.mock import Mock
        downloader=Mock(return_value='/cache/wetext')
        english=SimpleNamespace(normalize=lambda text,**kw:'twelve games')
        factory=Mock(return_value=english)
        module=SimpleNamespace(snapshot_download=downloader,Normalizer=factory)
        with patch.object(worker.importlib,'import_module',return_value=module), \
                patch.object(worker.os.path,'isfile',return_value=True):
            with worker._cached_normalizer():
                normalizer=module.Normalizer()
                self.assertEqual(normalizer.normalize('12 games'),'twelve games')
        self.assertTrue(all(c.kwargs.get('local_files_only') for c in downloader.call_args_list))
        self.assertEqual(factory.call_args.kwargs['lang'],'en')
        self.assertTrue(factory.call_args.kwargs['tagger_path'].endswith('en_tn_tagger.fst'))


if __name__ == '__main__':
    unittest.main()
