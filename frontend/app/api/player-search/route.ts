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
    const query = searchParams.get('query');

    if (!query || query.trim().length < 2) {
      return NextResponse.json({ people: [] });
    }

    // Hydrate currentTeam so we receive the full details (like currentTeam.id and currentTeam.name)
    const mlbApiUrl = `https://statsapi.mlb.com/api/v1/people/search?names=${encodeURIComponent(
      query
    )}&sportIds=1&hydrate=currentTeam`;

    const response = await fetch(mlbApiUrl, {
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      return NextResponse.json({ error: 'MLB Stats API 호출 실패' }, { status: 500 });
    }

    const data = await response.json();
    const people = data.people || [];

    // Map response structure and translate numeric MLB team ID to its abbreviation string
    const mappedPlayers = people.map((person: any) => {
      const numericTeamId = person.currentTeam?.id;
      const teamAbbr = numericTeamId ? (MLB_TEAM_ID_MAP[numericTeamId] ?? '') : '';

      return {
        id: person.id,
        fullName: person.fullName,
        teamId: teamAbbr,
        teamName: teamAbbr ? person.currentTeam.name : '팀 미정',
        position: person.primaryPosition?.abbreviation ?? 'N/A',
        pitchHand: person.pitchHand?.code ?? 'R',
        batSide: person.batSide?.code ?? 'R',
      };
    });

    return NextResponse.json({ people: mappedPlayers });
  } catch (error: any) {
    console.error('Player search proxy error:', error);
    return NextResponse.json(
      { error: error?.message || '선수 검색 중 서버 오류가 발생했습니다.' },
      { status: 500 }
    );
  }
}

