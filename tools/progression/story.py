#!/usr/bin/env python3
"""Re-edit cached gameplay into an accessible long film and narrated short.

python tools/progression/story.py --run progression_out/runs/OLD --out-dir progression_out/runs/NEW
No emulators are launched. Source recordings and the old run are never edited.
"""
import argparse
import hashlib
import math
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from tools.progression import short, video, narrate, speech, chart
import writer

PLAIN = ('Write for a middle-school viewer who knows games but knows nothing about AI. '
         'Use short everyday sentences and contractions. Say computer player, practice, '
         'saved version, and starting position. Never say checkpoint, policy, value network, '
         'temporal difference, epsilon, or training steps in spoken text or titles. '
         'Do not claim more practice always makes a better player. No URLs, Markdown, '
         'stage directions, forced jokes, or invented facts. The creator is choosing '
         'a computer opponent for a future human-versus-computer match.')
CLOSING = "Next video, I'm taking on this computer myself. Stay tuned for the match!"


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def select(panels, run, count=2):
    """Rank means on the identical full state suite; show its first state.

    Never select a different flattering starting position for each competitor.
    Ties are deterministic and disclosed; missing pairs/caps are refused.
    """
    groups = {}
    for p in panels:
        groups.setdefault(short.checkpoint_of(p), []).append(p)
    suite = list(run['states'])
    if not suite or len(suite) != len(set(suite)):
        raise ValueError('Need a nonempty suite of distinct starting positions')
    if len(groups) < count:
        raise ValueError('Need at least %d saved versions' % count)
    for key, rows in groups.items():
        if len(rows) != len(suite) or {p['state'] for p in rows} != set(suite):
            raise ValueError('Incomplete paired state suite for ' + key)
        for p in rows:
            if short.ending(p)['reason'] != 'game_over':
                raise ValueError('Story selection requires completed games: ' + p['key'])
    means = {k: sum(float(p['value']) for p in rows)/len(rows) for k, rows in groups.items()}
    manifest_order = [str(c[1]) for c in run['columns']]
    if set(manifest_order) != set(groups) or len(manifest_order) != len(groups):
        raise ValueError('Run column names do not match cached checkpoint identities')
    ranked = sorted(groups, key=lambda k: (-means[k],manifest_order.index(k)))
    chosen = set(ranked[:count])
    order = [str(c[1]) for c in run['columns'] if str(c[1]) in chosen]
    if len(order) != count:
        raise ValueError('Run column names do not match cached checkpoint identities')
    labels = ['Less practice', 'More practice'] if count == 2 else [
        'Less practice', 'More practice', 'Most practice']
    names = dict(zip(order, labels))
    shown = [dict(next(p for p in groups[k] if p['state'] == suite[0]), label=names[k])
             for k in order]
    results = [dict(column=names[k], value=p['value'], decisions=p['decisions'],
                    end_reason=p['end_reason'], state=p['state'], key=p['key'])
               for k in order for p in groups[k]]
    ties = [k for k in ranked if means[k] == means[ranked[0]]]
    return shown, results, dict(selected=order, labels=names, means=means,
        chosen_opponent=ranked[0], tied_best=ties, displayed_state=suite[0], states=suite,
        rule='Top mean results across the full common suite; first listed state shown. '
             'Practice labels are relative to the selected versions in manifest order.')


def font(size):
    for name in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def source_windows(panels, lengths):
    """Non-overlapping windows on one clock, preserving the first loss if feasible."""
    end = max(short.ending(p)['time'] for p in panels)
    speed = max(1.,min(4.,end/sum(lengths)))
    widths = [d*speed for d in lengths]
    spare = max(0,end-sum(widths))
    starts, cursor = [],0.
    for i,width in enumerate(widths):
        starts.append(cursor + spare*i/max(1,len(widths)-1))
        cursor += width
    if len(starts)>=3:
        i = len(starts)-2
        first_loss = min(short.ending(p)['time'] for p in panels)
        low = starts[i-1]+widths[i-1]
        high = starts[i+1]-widths[i]
        starts[i] = max(low,min(high,first_loss-widths[i]+speed))
    return starts,speed


def plate(draw, text, y, width, size=36):
    face = ImageFont.truetype(short.OVERLAY_FONT_FILE, size)
    # Fit by measured pixels; never silently truncate the spoken caption.
    lines, line = [], ''
    for word in text.split():
        trial = (line + ' ' + word).strip()
        if line and draw.textlength(trial, font=face) > width - 90:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    for line in lines:
        x = (width - draw.textlength(line, font=face))/2
        box = draw.textbbox((x, y), line, font=face)
        draw.rectangle((box[0]-10, box[1]-8, box[2]+10, box[3]+8), fill='white')
        draw.text((x, y), line, font=face, fill='black')
        y += size + 18
    return y


def result_card(results, names, chosen, unit, path, tied=False):
    """One readable takeaway; detailed numbers stay in results.json."""
    stats = {name: chart.summarise([r for r in results if r['column'] == name])
             for name in names}
    image = Image.new('RGB', (1920,1080), '#090b10')
    draw = ImageDraw.Draw(image)
    plate(draw, 'MY NEXT OPPONENT', 55, 1920, 56)
    plate(draw, chosen, 165, 1920, 50)
    label = 'Average %s per game' % unit
    plate(draw, label, 280, 1920, 36)
    ceiling = max([abs(s['mean']) for s in stats.values()] + [1.0])
    for i,name in enumerate(names):
        y = 410+i*125
        # All lettering has an opaque plate, including the chart's labels.
        f = font(40)
        text = '%s: %.1f' % (name,stats[name]['mean'])
        box = draw.textbbox((120,y),text,font=f)
        draw.rectangle((box[0]-10,box[1]-8,box[2]+10,box[3]+8),fill='white')
        draw.text((120,y),text,font=f,fill='black')
        bar = max(1,int(800*max(0,stats[name]['mean'])/ceiling))
        draw.rectangle((950,y,950+bar,y+60),fill=chart.GOLD if name==chosen else chart.BAR)
    chart.crown(draw,960,890,70)
    note = ('Tied average; this is my chosen opponent.' if tied else
            'Same %d starting positions for each player.' % stats[names[0]]['n'])
    plate(draw,note,950,1920,30)
    image.save(path)
    return dict(path=str(path),stats=stats)


def render_scene(panels, path, duration, start=0, speed=1, portrait=False,
                 caption='', heading='', crown=False):
    """Fixed board positions, shared source time, captions on opaque plates."""
    width, height = (1080, 1920) if portrait else (1920, 1080)
    n = len(panels)
    cell_w = ((width - 80)//n)//2*2  # even dimensions for YUV420, including three boards
    cell_h = 1000 if portrait else 660
    top = 400 if portrait else 145
    overlay = str(Path(path).with_suffix('.png'))
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    plate(draw, heading, 110 if portrait else 25, width, 40)
    for i, p in enumerate(panels):
        label = p['label']
        f = font(28 if portrait else 34)
        x = 40 + i*cell_w + (cell_w-draw.textlength(label,font=f))/2
        draw.rectangle((x-8, top-55, x+draw.textlength(label,font=f)+8, top-5), fill='white')
        draw.text((x, top-52), label, font=f, fill='black')
    plate(draw, caption, 1510 if portrait else 865, width, 38 if portrait else 34)
    if crown:
        chart.crown(draw, width/2 if portrait else width/2-280,
                    315 if portrait else 130, 65)
    image.save(overlay)
    inputs, filters = [], []
    for i, p in enumerate(panels):
        source_len = float(p['seconds'])
        seek = min(start, max(0, source_len - .1))
        inputs += ['-ss', str(seek), '-i', p['path']]
        dead_at = max(0, (short.ending(p)['time']-start)/speed)
        filters.append('[%d:v]setpts=(PTS-STARTPTS)/%g,tpad=stop_mode=clone:stop_duration=%g,'
            'trim=duration=%g,scale=%d:%d:force_original_aspect_ratio=decrease:flags=neighbor,'
            'pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=30,'
            "eq=brightness=-0.15:enable='gte(t,%g)'[p%d]" %
            (i,speed,duration,duration,cell_w,cell_h,cell_w,cell_h,dead_at,i))
    chain = ''.join('[p%d]' % i for i in range(n))
    filters.append(chain + ('hstack=inputs=%d' % n if n > 1 else 'null') + '[boards]')
    filters.append('[boards]pad=%d:%d:40:%d:color=0x090b10[bg]' % (width,height,top))
    inputs += ['-loop','1','-framerate','30','-i',overlay]
    filters.append('[bg][%d:v]overlay=0:0:shortest=1[art]' % n)
    stage = '[art]'
    # Render the source clock every second; jumps are explicit at every cut.
    for tick in range(int(duration)+1):
        target = '[clock%d]' % tick
        message = 'Game time ' + video._clock(start + tick*speed)
        filters.append(short._enabled_text(stage,target,message,tick,tick+1,
                                            25, height-60,22))
        stage = target
    for i, p in enumerate(panels):
        target = '[end%d]' % i
        death = max(0,(short.ending(p)['time']-start)/speed)
        filters.append(short._enabled_text(stage,target,'GAME OVER',death,duration,
                         str(55+i*cell_w),top+cell_h-60,24))
        stage = target
    frames = round(duration*30)
    filters.append(stage+'tpad=stop_mode=clone:stop_duration=1,trim=end_frame=%d,'
                   'setpts=PTS-STARTPTS[finished]' % frames)
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error',*inputs,
        '-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[finished]',
        '-an','-frames:v',str(frames),'-c:v','libx264','-preset','veryfast','-crf','20',
        '-pix_fmt','yuv420p','-r','30',path],check=True)
    return path


def concat(paths, dest):
    listing = Path(dest).with_suffix('.concat.txt')
    listing.write_text(''.join("file '%s'\n" % Path(p).resolve().as_posix().replace("'", "'\\''")
                               for p in paths), encoding='utf-8')
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-f','concat','-safe','0',
                    '-i',str(listing),'-c','copy','-movflags','+faststart',dest],check=True)


def prompt_script(facts, seconds, short_form=False):
    if short_form:
        brief = ('One creator voice. Return exactly three body entries, who=1. '
                 'First: introduce the contest. Second: explain that these are saved '
                 'versions of the same computer with different amounts of practice, '
                 'tested from the same starting positions. Third: say that more '
                 'practice does not guarantee a better player. Each entry 24-29 words. '
                 'The card field must be empty. Closing <=20 words: name the chosen '
                 'opponent based on ALL the games and invite viewers to the next match.')
    else:
        brief = ('A warm two-host conversation: host 1 built the player, host 2 is a '
                 'curious gaming friend. They respond directly, ask short questions, '
                 'and use occasional gentle humour. No lecture or game-history detour. '
                 'Exactly six body turns alternating hosts, each 25-35 words. Explain '
                 'the contest and learning through practice without technical claims. '
                 'Do not reveal results during the body. Card: exactly one who=1 turn, '
                 '<=12 words, naming the best average version WITHOUT reading numbers. Closing: who=1, <=18 words, '
                 'explicitly say next video I play it myself and stay tuned for the match.')
    return PLAIN + '\n' + brief + '\nTarget seconds: ' + str(seconds) + '\nFACTS:\n' + json.dumps(facts)


def write_script(facts, seconds, out, prefix, options, short_form=False, resume=False):
    prompt = prompt_script(facts, seconds, short_form)
    (out/'prompts'/('%s.txt' % prefix)).write_text(prompt,encoding='utf-8')
    save(out/'prompts'/(prefix+'_schema.json'),narrate.PODCAST_SCHEMA)
    script_path = out/(prefix+'.json')
    if resume and script_path.exists():
        data = json.loads(script_path.read_text(encoding='utf-8'))
        validate_script(data, short_form)
        return data
    data = writer.write(prompt,narrate.PODCAST_SCHEMA,**options)
    if not data:
        raise RuntimeError('Writer failed; saved prompt: ' + prefix)
    def clean(item):
        return (writer.clean_for_speech(item['text']), 2 if item.get('who') == 2 else 1)
    result = dict(body=[clean(t) for t in data['body']],card=[clean(t) for t in data['card']],
                  closing=clean(data['closing']))
    # An explicit invitation is a product requirement, independent of model compliance.
    ending = result['closing'][0]
    if 'stay tuned' not in ending.lower() or 'next' not in ending.lower():
        ending = (facts['chosen_opponent'] + ' is my next opponent. Stay tuned for our match!'
                  if short_form else CLOSING)
    result['closing'] = (ending,1)
    expected = 3 if short_form else 6
    if len(result['body']) != expected:
        raise RuntimeError('Writer returned %d body turns; expected %d. Rerun; prompt saved.' %
                           (len(result['body']),expected))
    if short_form:
        result['body'] = [(t,1) for t,_ in result['body']]
        result['card'] = []
    elif len(result['card']) != 1:
        raise RuntimeError('Writer must return one short result sentence')
    else:
        result['body'] = [(text,1+i%2) for i,(text,_) in enumerate(result['body'])]
    validate_script(result, short_form)
    save(script_path,result)
    return result


def validate_script(script, short_form=False):
    expected = 3 if short_form else 6
    if len(script['body']) != expected or len(script['card']) != (0 if short_form else 1):
        raise ValueError('Saved script has the wrong number of turns')
    for text, who in script['body'] + script['card'] + [script['closing']]:
        if not text.strip() or who not in (1, 2) or len(text.split()) > 45:
            raise ValueError('Each turn needs 1-45 words and host 1 or 2')
    closing = script['closing'][0].lower()
    if 'stay tuned' not in closing or 'next' not in closing:
        raise ValueError('Closing must mention next video/match and stay tuned')


def make_film(panels, winner, script, out, prefix, seconds, voices, model, chart_path=None):
    portrait = prefix == 'short'
    turns = script['body'] + script['card'] + [script['closing']]
    refs = [voices[(who-1) if not portrait else 0] for _,who in turns]
    clips = speech.speak([t for t,_ in turns],str(out/(prefix+'_speech')),backend=model,voice=refs)
    if len(clips) != len(turns):
        raise RuntimeError('Speech backend omitted a turn; refusing misaligned narration')
    body_n = len(script['body'])
    body = clips[:body_n]
    result = clips[body_n:-1]
    closing = clips[-1]
    # Guaranteed closing reservation; nothing is silently dropped or accelerated.
    closing_frames = math.ceil(max(6.0,closing[1]+1)*30)
    closing_time = closing_frames/30
    card_time = 7.0 if chart_path else 0
    if result and result[0][1] > card_time-.8:
        raise RuntimeError('Result sentence exceeds the seven-second card; shorten it in %s.json' % prefix)
    available_frames = round(seconds*30)-round(card_time*30)-closing_frames
    body_frames = [math.ceil((duration+.7)*30) for _,duration in body]
    if sum(body_frames) > available_frames:
        raise RuntimeError('Speech exceeds %gs budget. Shorten %s.json; audio is preserved.' % (seconds,prefix))
    # In the podcast, don't put long silent gaps between a question and its
    # answer. Breathing room belongs after each two-turn exchange.
    pause_slots = list(range(len(body_frames))) if portrait else list(range(1,len(body_frames),2))
    slack,extra = divmod(available_frames-sum(body_frames),len(pause_slots))
    for i,slot in enumerate(pause_slots):
        body_frames[slot] += slack+(i<extra)
    lengths = [frames/30 for frames in body_frames]
    # Each scene owns a measured speech slot. Sample the shared timeline evenly;
    # both boards always use the identical source window and remain in place.
    if portrait:
        starts,playback_speed = source_windows(panels,lengths)
    else:
        # Only three source jumps; inside each exchange playback is continuous.
        pair_lengths = [sum(lengths[i:i+2]) for i in range(0,len(lengths),2)]
        pair_starts,playback_speed = source_windows(panels,pair_lengths)
        starts = [t for i,start in enumerate(pair_starts)
                  for t in (start,start+lengths[2*i]*playback_speed)]
    parts,placed,cursor = [],[],0.0
    for i, ((audio,dur),length,start) in enumerate(zip(body,lengths,starts)):
        opening = i==0 if portrait else i<2
        heading = 'Which computer should I play?' if opening else 'Later in the same game...'
        parts.append(render_scene(panels,str(out/('%s_scene%d.mp4'%(prefix,i))),length,start,
                                  playback_speed,portrait,script['body'][i][0],heading))
        placed.append((cursor+.25,cursor+.25+dur,audio,False))
        cursor += length
    if chart_path:
        card_mp4 = str(out/'result_card.mp4')
        # Use the same frame rate as the gameplay for concat.
        subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-loop','1','-framerate','30','-i',chart_path,
                        '-t',str(card_time),'-r','30','-c:v','libx264','-pix_fmt','yuv420p',
                        card_mp4],check=True)
        parts.append(card_mp4)
        if result:
            placed.append((cursor+.3,cursor+.3+result[0][1],result[0][0],True))
        cursor += card_time
    # Moving footage replay, explicitly labelled, under the complete invitation.
    start = max(0, min(winner['seconds']*.5,winner['seconds']-closing_time))
    parts.append(render_scene([winner],str(out/(prefix+'_outro.mp4')),closing_time,start,
                              1,portrait,script['closing'][0],
                              'MY NEXT OPPONENT - REPLAY',True))
    placed.append((cursor+.25,cursor+.25+closing[1],closing[0],False))
    silent = str(out/(prefix+'_silent.mp4'))
    concat(parts,silent)
    wav = str(out/(prefix+'_narration.wav'))
    narrate.build_track(placed,seconds,wav)
    dest = str(out/('short_9x16.mp4' if portrait else 'narrated.mp4'))
    narrate.mux(silent,wav,dest)
    save(out/(prefix+'_timeline.json'),dict(placed=placed,source_starts=starts,
                                           lengths=lengths,seconds=seconds,speed=playback_speed))
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--count',type=int,choices=(2,3),default=2)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--short-seconds',type=float,default=60)
    parser.add_argument('--voice-model',default=speech.DEFAULT)
    parser.add_argument('--voice-name',default='voice_references/host1.wav,voice_references/host2.wav')
    parser.add_argument('--writer',choices=writer.BACKENDS,default='chatgpt')
    parser.add_argument('--resume',action='store_true',help='Reuse editable scripts after a failed render')
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if args.seconds < 45 or args.short_seconds < 30:
        raise SystemExit('Narrated package needs at least 45 seconds long / 30 seconds short.')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit('Choose an empty output folder; existing packages are preserved.')
    speech.validate_voice(args.voice_model,args.voice_name,podcast=True)
    if not speech.available(args.voice_model):
        raise SystemExit('Speech backend is unavailable: '+args.voice_model)
    panels,run,_ = short.panels_from_run(args.run)
    shown,results,selection = select(panels,run,args.count)
    # Resume only this exact source/settings. Cached movies are read, never modified.
    identity = dict(run=run, count=args.count, seconds=args.seconds, short_seconds=args.short_seconds,
                    voice_model=args.voice_model, voice_name=args.voice_name, writer=args.writer,
                    sources=[dict(key=p['key'],size=Path(p['path']).stat().st_size,
                                  modified=Path(p['path']).stat().st_mtime_ns,
                                  timeline=p['timeline'],value=p['value']) for p in panels])
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if args.resume:
        saved = json.loads((out/'package.json').read_text(encoding='utf-8'))
        if saved['signature'] != signature:
            raise SystemExit('Source/settings changed. Choose a new output folder.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'prompts').mkdir(exist_ok=True)
    (out/'prompts'/'writer_system.txt').write_text(writer.VOICE,encoding='utf-8')
    save(out/'package.json',dict(signature=signature,settings=identity))
    save(out/'selection.json',selection)
    save(out/'results.json',results)
    save(out/'source_run.json',run)
    winner = next(p for p in shown if short.checkpoint_of(p)==selection['chosen_opponent'])
    unit = str((shown[0].get('stats') or [['points',0]])[0][0]).lower()
    card = result_card(results,[p['label'] for p in shown],winner['label'],unit,
                       out/'chart.png',len(selection['tied_best'])>1)
    facts = dict(game=run.get('title') or run['game'],results=card['stats'],
                 chosen_opponent=winner['label'],states=len(selection['states']),
                 shown_start=selection['displayed_state'],
                 caveat='We show one shared start. Opponent chosen using the average across all starts.',
                 tied_best=[selection['labels'].get(k,'another saved version')
                            for k in selection['tied_best']],
                 tie_rule='If tied, say chosen opponent, not a sole winner.')
    options = short._writer_options(run,args.writer)
    long_script = write_script(facts,args.seconds,out,'long_script',options,resume=args.resume)
    short_script = write_script(facts,args.short_seconds,out,'short_script',options,True,args.resume)
    # Generate and preserve all four upload forms, with the same audience brief.
    schema = dict(type='object',properties={k:short.PLATFORM_SCHEMA for k in
                 ('youtube_long','youtube_short','instagram','tiktok')},
                 required=['youtube_long','youtube_short','instagram','tiktok'],additionalProperties=False)
    prompt = PLAIN+'\nWrite titles, descriptions and tags for the four uploads. Both videos '
    prompt += 'have narration and captions. Explain the same-start comparison and next human match. '
    prompt += 'The shown game is an example; the selection uses all games. Facts: '+json.dumps(facts)
    (out/'prompts'/'uploads.txt').write_text(prompt,encoding='utf-8')
    save(out/'prompts'/'uploads_schema.json',schema)
    uploads = (json.loads((out/'uploads.json').read_text(encoding='utf-8'))
               if args.resume and (out/'uploads.json').exists() else writer.write(prompt,schema,**options))
    if not uploads:
        raise RuntimeError('Upload copy writer failed; prompt saved')
    save(out/'uploads.json',uploads)
    voices = [v.strip() for v in args.voice_name.split(',')]
    make_film(shown,winner,long_script,out,'long',args.seconds,voices,args.voice_model,card['path'])
    make_film(shown,winner,short_script,out,'short',args.short_seconds,voices,args.voice_model)
    brief = ['# Upload package\n', 'Selection details and exact training versions: selection.json\n']
    for platform,copy in uploads.items():
        filename = 'narrated.mp4' if platform=='youtube_long' else 'short_9x16.mp4'
        brief += ['## '+platform+'\n', 'File: '+filename+'\n',copy['title']+'\n',
                  copy['description']+'\n',', '.join(copy['tags'])+'\n']
        (out/(platform+'.md')).write_text('\n'.join(brief[-5:]),encoding='utf-8')
    (out/'UPLOAD_BRIEF.md').write_text('\n'.join(brief),encoding='utf-8')
    print('Completed package:',out)


if __name__=='__main__':
    main()
