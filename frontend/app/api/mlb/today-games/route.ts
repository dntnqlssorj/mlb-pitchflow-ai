import { NextResponse } from 'next/server';

function getETDate(offsetDays = 0): string {
  const now = new Date();
  now.setDate(now.getDate() + offsetDays);
  const nyDate = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(now);
  const [month, day, year] = nyDate.split('/');
  return `${year}-${month}-${day}`;
}

export async function GET() {
  try {
    const today = getETDate(0);
    const yesterday = getETDate(-1);

    const [todayRes, yesterdayRes] = await Promise.all([
      fetch(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${today}`, { next: { revalidate: 15 } }),
      fetch(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${yesterday}`, { next: { revalidate: 15 } }),
    ]);

    const [todayData, yesterdayData] = await Promise.all([
      todayRes.ok ? todayRes.json() : { dates: [] },
      yesterdayRes.ok ? yesterdayRes.json() : { dates: [] },
    ]);

    const todayGames = todayData.dates?.[0]?.games || [];
    const yesterdayGames = yesterdayData.dates?.[0]?.games || [];
    const allGames = [...todayGames, ...yesterdayGames];

    // Live 진행 중 경기만 반환 (Preview/Final 제외)
    const seenPks = new Set<number>();
    const activeGames = allGames
      .filter((g: any) => {
        const isLive = g.status.abstractGameState === 'Live';
        const isDupe = seenPks.has(g.gamePk);
        if (isLive && !isDupe) {
          seenPks.add(g.gamePk);
          return true;
        }
        return false;
      })
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
