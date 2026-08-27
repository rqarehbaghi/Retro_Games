import { RetroGame } from '../types';

export const RETRO_GAMES: RetroGame[] = [
  {
    id: 'MortalKombatII-Genesis',
    name: 'Mortal Kombat II',
    console: 'Genesis',
    genre: 'fighting',
    twoPlayerSupport: true,
    supportedModes: ['versus', 'autonomous'],
    description: 'Iconic 1v1 fighting game with special moves, fatalities, and frame-accurate combos.',
    defaultState: 'Level1.LiuKang',
    stateOptions: ['Level1.LiuKang', 'Level1.SubZero', 'Level1.Scorpion', 'KombatZone.PitII'],
    recommendedCombos: [
      [],
      ['RIGHT'],
      ['LEFT'],
      ['UP'],
      ['DOWN'],
      ['A'],
      ['B'],
      ['C'],
      ['X'],
      ['Y'],
      ['Z'],
      ['DOWN', 'A'],
      ['DOWN', 'B'],
      ['RIGHT', 'A'],
      ['RIGHT', 'B'],
      ['DOWN', 'RIGHT', 'A'],
    ],
    aiDifficultyPresets: [
      { iter: 1, title: 'Iter 1: Random Flailer', description: 'Untouched random CNN network. Spams crouch and punches air.', winRateVsHuman: 3, reactionFrames: 30 },
      { iter: 100, title: 'Iter 100: Novice Fighter', description: 'Learned to walk forward and throw light punches; blocks 10% of sweeps.', winRateVsHuman: 28, reactionFrames: 18 },
      { iter: 500, title: 'Iter 500: Frame-Trapper', description: 'Consistently buffers projectiles and punishes jump-ins with uppercuts.', winRateVsHuman: 68, reactionFrames: 6 },
      { iter: 2000, title: 'Iter 2000: Superhuman Boss', description: 'Frame-perfect anti-air reactions, unblockable mixups, and 100% combo execution.', winRateVsHuman: 94, reactionFrames: 1 },
    ],
    viralHooks: [
      'I Trained an AI on Mortal Kombat for 10,000 Iterations... and Regretted It',
      'Can a Human Beat an AI That Reads Inputs in 1 Frame?',
      'Teaching an AI to Fatality Me: Day 1 vs Day 30'
    ]
  },
  {
    id: 'StreetFighterIISNES',
    name: 'Street Fighter II Turbo',
    console: 'SNES',
    genre: 'fighting',
    twoPlayerSupport: true,
    supportedModes: ['versus', 'autonomous'],
    description: 'The golden standard of competitive 2D fighters with 6-button depth and zoning tactics.',
    defaultState: 'RyuVsKen.Bicycle',
    stateOptions: ['RyuVsKen.Bicycle', 'ChunLiVsGuile', 'BisonChampionship'],
    recommendedCombos: [
      [],
      ['RIGHT'],
      ['LEFT'],
      ['DOWN'],
      ['UP'],
      ['Y'],
      ['X'],
      ['L'],
      ['B'],
      ['A'],
      ['R'],
      ['DOWN', 'RIGHT', 'Y'],
      ['RIGHT', 'DOWN', 'RIGHT', 'X']
    ],
    aiDifficultyPresets: [
      { iter: 1, title: 'Iter 1: Button Masher', description: 'Random twitching, walks into corners.', winRateVsHuman: 2, reactionFrames: 32 },
      { iter: 150, title: 'Iter 150: Fireball Spammer', description: 'Discovered Hadouken inputs; repeats fireball zoning.', winRateVsHuman: 35, reactionFrames: 14 },
      { iter: 800, title: 'Iter 800: Shoryuken Master', description: 'Invincible dragon punch anti-air timing on every single jump.', winRateVsHuman: 74, reactionFrames: 4 },
      { iter: 3000, title: 'Iter 3000: Evo Champion AI', description: 'TAS-level parry timings and instant corner vortex traps.', winRateVsHuman: 97, reactionFrames: 1 }
    ],
    viralHooks: [
      'AI vs Pro Street Fighter Player: The Ultimate Showdown',
      'This AI Discovered a 100% Unblockable Combo in SF2',
      'How Many Hours to Beat My Own AI Creation?'
    ]
  },
  {
    id: 'SuperMarioBros3-Nes-v0',
    name: 'Super Mario Bros 3',
    console: 'NES',
    genre: 'platformer',
    twoPlayerSupport: false,
    supportedModes: ['race', 'autonomous'],
    description: 'Acclaimed NES platformer featuring power-ups, momentum physics, and flying mechanics.',
    defaultState: 'SuperMarioBros3-Nes-v0.state',
    stateOptions: ['SuperMarioBros3-Nes-v0.state', 'World1-1', 'World1-Fortress', 'World8-Airship'],
    recommendedCombos: [
      [],
      ['RIGHT'],
      ['LEFT'],
      ['RIGHT', 'A'],
      ['RIGHT', 'B'],
      ['RIGHT', 'A', 'B'],
      ['LEFT', 'A', 'B'],
      ['A'],
      ['B'],
      ['DOWN']
    ],
    aiDifficultyPresets: [
      { iter: 1, title: 'Iter 1: Random Flailing', description: 'Runs left into initial Goombas or jumps in place.', winRateVsHuman: 0, reactionFrames: 30 },
      { iter: 100, title: 'Iter 100: Novice Runner', description: 'Learned to hold Right and small hop over flat obstacles.', winRateVsHuman: 15, reactionFrames: 16 },
      { iter: 500, title: 'Iter 500: Speedrunner', description: 'Maintains P-Wing sprint speed, bouncing off enemies across chasms.', winRateVsHuman: 82, reactionFrames: 3 },
      { iter: 2500, title: 'Iter 2500: TAS God', description: 'Frame-perfect sub-pixel velocity optimization without taking 1 damage.', winRateVsHuman: 99, reactionFrames: 1 }
    ],
    viralHooks: [
      'Can You Beat an AI in a Mario 3 Speedrun Race?',
      'AI Learns to Beat World 1 Without Touching the Ground',
      'The Exact Moment My AI Figured Out Mario Momentum'
    ]
  },
  {
    id: 'Contra-Nes',
    name: 'Contra',
    console: 'NES',
    genre: 'shooter',
    twoPlayerSupport: true,
    supportedModes: ['coop', 'versus', 'race', 'autonomous'],
    description: 'Legendary 2-player run-and-gun military action. True test of teamwork or rivalry!',
    defaultState: '1Player.Jungle',
    stateOptions: ['1Player.Jungle', '2Player.Jungle', 'Level3.Waterfall', 'Level8.AlienLair'],
    recommendedCombos: [
      ['RIGHT', 'B'],
      ['RIGHT', 'A', 'B'],
      ['UP', 'B'],
      ['UP', 'RIGHT', 'B'],
      ['DOWN', 'B'],
      ['A', 'B'],
      ['LEFT', 'B']
    ],
    aiDifficultyPresets: [
      { iter: 1, title: 'Iter 1: Friendly Fire Hazard', description: 'Shoots ceiling, runs backward into turrets.', winRateVsHuman: 5, reactionFrames: 30 },
      { iter: 200, title: 'Iter 200: Cover Shooter', description: 'Prone dodges bullets and provides sustained diagonal fire.', winRateVsHuman: 45, reactionFrames: 10 },
      { iter: 1000, title: 'Iter 1000: Ultimate Co-op Partner', description: 'Synchronized screen-clearing spread gun angles, carries human through boss phases.', winRateVsHuman: 90, reactionFrames: 2 }
    ],
    viralHooks: [
      'I Trained an AI to Be My Player 2 in Contra: 1-Life Challenge',
      'Can an AI Carry Me Through the Hardest Retro Game?',
      'When Your AI Co-op Partner Is Better Than You'
    ]
  },
  {
    id: 'StreetsOfRage2-Genesis',
    name: 'Streets of Rage 2',
    console: 'Genesis',
    genre: 'beatemup',
    twoPlayerSupport: true,
    supportedModes: ['coop', 'versus', 'autonomous'],
    description: 'Peak side-scrolling beat em up with Axel, Blaze, and devastating grand upper specials.',
    defaultState: 'Stage1.AxelBlaze',
    stateOptions: ['Stage1.AxelBlaze', 'Stage2.Bridge', 'Stage8.MrX'],
    recommendedCombos: [
      ['RIGHT'],
      ['LEFT'],
      ['UP'],
      ['DOWN'],
      ['B'],
      ['C'],
      ['A'],
      ['RIGHT', 'RIGHT', 'B'],
      ['FORWARD', 'A']
    ],
    aiDifficultyPresets: [
      { iter: 1, title: 'Iter 1: Alley Victim', description: 'Gets trapped by basic Galsia thugs.', winRateVsHuman: 2, reactionFrames: 30 },
      { iter: 300, title: 'Iter 300: Brawler', description: 'Performs combo finishers and throws enemies into groups.', winRateVsHuman: 55, reactionFrames: 8 },
      { iter: 1200, title: 'Iter 1200: Grand Upper God', description: 'Infinite juggle setups and zero-damage boss dispatches.', winRateVsHuman: 92, reactionFrames: 2 }
    ],
    viralHooks: [
      'Beating Streets of Rage 2 with an AI Partner on Mania Difficulty',
      'Human & AI vs 100 Street Brawlers',
      'My AI Learned to Steal All My Health Pickups'
    ]
  }
];
