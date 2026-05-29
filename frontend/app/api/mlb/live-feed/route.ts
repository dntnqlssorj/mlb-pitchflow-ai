import { NextRequest, NextResponse } from 'next/server';

const MLB_TEAM_ID_MAP: Record<number, string> = {
  133: 'OAK', 134: 'PIT', 135: 'SD', 136: 'SEA',
  137: 'SF', 138: 'STL', 139: 'TB', 140: 'TEX',
  141: 'TOR', 142: 'MIN', 143: 'PHI', 144: 'ATL',
  145: 'CWS', 146: 'MIA', 147: 'NYY', 158: 'MIL',
  108: 'LAA', 109: 'ARI', 110: 'BAL', 111: 'BOS',
  112: 'CHC', 113: 'CIN', 114: 'CLE', 115: 'COL',
  116: 'DET', 117: 'HOU', 118: 'KC', 119: 'LAD',
  120: 'WSH', 121: 'NYM'
};

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const gamePk = searchParams.get('gamePk');

    if (!gamePk) {
      return NextResponse.json({ error: 'gamePk parameter is required' }, { status: 400 });
    }

    const url = `https://statsapi.mlb.com/api/v1.1/game/${gamePk}/feed/live`;
    const res = await fetch(url, { next: { revalidate: 5 } });

    if (!res.ok) {
      throw new Error(`MLB Stats API live-feed returned status: ${res.status}`);
    }

    const data = await res.json();
    const liveData = data.liveData;
    const gameData = data.gameData;

    if (!liveData || !liveData.linescore) {
      return NextResponse.json(null);
    }

    const linescore = liveData.linescore;
    const currentPlay = liveData.plays?.currentPlay;
    const matchup = currentPlay?.matchup;

    // Determine Inning and Side
    const inning = linescore.currentInning || 1;
    const isTopInning = linescore.isTopInning !== false; // default true if missing

    // Extract Pitcher and Batter details
    const pitcherId = linescore.defense?.pitcher?.id || matchup?.pitcher?.id || 0;
    const pitcherName = linescore.defense?.pitcher?.fullName || matchup?.pitcher?.fullName || 'Pitcher';
    
    const batterId = linescore.offense?.batter?.id || matchup?.batter?.id || 0;
    const batterName = linescore.offense?.batter?.fullName || matchup?.batter?.fullName || 'Batter';

    // Track teams
    const awayTeamIdVal = gameData?.teams?.away?.id;
    const homeTeamIdVal = gameData?.teams?.home?.id;

    const awayTeamAbbr = MLB_TEAM_ID_MAP[awayTeamIdVal] || '';
    const homeTeamAbbr = MLB_TEAM_ID_MAP[homeTeamIdVal] || '';

    // Assign teamIds based on half-inning logic:
    // Top Inning (초): Batter is Away, Pitcher is Home
    // Bottom Inning (말): Batter is Home, Pitcher is Away
    const batterTeamId = isTopInning ? awayTeamAbbr : homeTeamAbbr;
    const pitcherTeamId = isTopInning ? homeTeamAbbr : awayTeamAbbr;

    // Resolve hands from matchup
    const pitcherHand: 'R' | 'L' = matchup?.pitchHand?.code === 'L' ? 'L' : 'R';
    const batterSide: 'R' | 'L' = matchup?.batSide?.code === 'L' ? 'L' : 'R';

    // Parse runners (Linescore contains first, second, third objects if bases are occupied)
    const on1b = !!linescore.offense?.first;
    const on2b = !!linescore.offense?.second;
    const on3b = !!linescore.offense?.third;

    // Parse last pitch
    let lastPitch = null;
    const playEvents = currentPlay?.playEvents || [];
    const pitches = playEvents.filter((ev: any) => ev.isPitch === true);
    if (pitches.length > 0) {
      const lastPitchEvent = pitches[pitches.length - 1];
      const code = lastPitchEvent.details?.type?.code || '';
      const desc = lastPitchEvent.details?.description || 'Pitch thrown';
      lastPitch = {
        type: code,
        description: desc
      };
    }

    const situation = {
      inning,
      isTopInning,
      balls: linescore.balls ?? 0,
      strikes: linescore.strikes ?? 0,
      outs: linescore.outs ?? 0,
      awayScore: linescore.teams?.away?.runs ?? 0,
      homeScore: linescore.teams?.home?.runs ?? 0,
      currentPitcher: { id: pitcherId, name: pitcherName },
      currentBatter: { id: batterId, name: batterName },
      on1b,
      on2b,
      on3b,
      pitcherTeamId,
      batterTeamId,
      pitcherHand,
      batterSide,
      lastPitch
    };

    return NextResponse.json(situation);
  } catch (error: any) {
    console.error('Error in live-feed API:', error);
    return NextResponse.json(
      { error: error?.message || 'Failed to retrieve live game feed.' },
      { status: 500 }
    );
  }
}
