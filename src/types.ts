export type GameGenre = 'fighting' | 'platformer' | 'shooter' | 'beatemup' | 'racing' | 'sports';

export type GameMode = 'versus' | 'coop' | 'race' | 'autonomous';

export interface RetroGame {
  id: string;
  name: string;
  console: 'NES' | 'Genesis' | 'SNES' | 'Arcade';
  genre: GameGenre;
  twoPlayerSupport: boolean;
  supportedModes: GameMode[];
  description: string;
  defaultState?: string;
  stateOptions: string[];
  recommendedCombos: string[][];
  aiDifficultyPresets: {
    iter: number;
    title: string;
    description: string;
    winRateVsHuman: number;
    reactionFrames: number;
  }[];
  viralHooks: string[];
}

export interface MatchState {
  gameId: string;
  mode: GameMode;
  stateName?: string;
  p1Name: string;
  p1Health: number;
  p2Name: string;
  p2Health: number;
  p1Score: number;
  p2Score: number;
  p1Wins: number;
  p2Wins: number;
  currentRound: number;
  timer: number;
  isPlaying: boolean;
  isPaused: boolean;
  matchOver: boolean;
  winner: 'p1' | 'p2' | 'tie' | null;
  p1LastAction: string;
  p2LastAction: string;
  aiProbabilities: { action: string; prob: number }[];
  p1CombosCount: number;
  p2CombosCount: number;
  stepCount: number;
}

export interface TrainingMetric {
  iteration: number;
  timesteps: number;
  meanReward: number;
  fps: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  notes: string;
}

export interface VideoSettings {
  aspectRatio: '16:9' | '9:16' | '1:1';
  resolution: '1080p' | '4k' | '720p';
  clipSeconds: number;
  selectedIterations: number[];
  overlayStyle: 'arcade' | 'minimal' | 'retro-glow' | 'cyberpunk';
  showIterationBadge: boolean;
  showHealthBars: boolean;
  showTimer: boolean;
  crtFilter: boolean;
  watermarkText: string;
  customHeadline: string;
}
