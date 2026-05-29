import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Determine the current date in US Eastern Time (New York) to align with MLB schedule timezone
    const nyDate = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(new Date());

    const [month, day, year] = nyDate.split('/');
    const today = `${year}-${month}-${day}`;

    const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${today}`;
    const res = await fetch(url, { next: { revalidate: 15 } });

    if (!res.ok) {
      throw new Error(`MLB Stats API schedule returned status: ${res.status}`);
    }

    const data = await res.json();
    const gamesList = data.dates?.[0]?.games || [];

    // Filter out games that have already finished (Final status)
    const activeGames = gamesList
      .filter((g: any) => g.status.abstractGameState !== 'Final')
      .map((g: any) => ({
        gamePk: g.gamePk,
        awayTeam: g.teams.away.team.name,
        homeTeam: g.teams.home.team.name,
        venue: g.venue.name,
        status: g.status.abstractGameState,
      }));

    return NextResponse.json(activeGames);
  } catch (error: any) {
    console.error('Error in today-games API:', error);
    return NextResponse.json(
      { error: error?.message || 'Failed to retrieve today\'s games.' },
      { status: 500 }
    );
  }
}
