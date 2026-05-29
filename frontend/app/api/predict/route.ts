import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const webhookUrl = process.env.N8N_CUSTOM_WEBHOOK_URL;

    if (!webhookUrl || webhookUrl === '여기에_입력') {
      return NextResponse.json({ error: 'Webhook URL 미설정' }, { status: 500 });
    }

    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000)
    });

    if (!response.ok) {
      return NextResponse.json({ error: 'n8n 호출 실패' }, { status: 500 });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Prediction API error:', error);
    return NextResponse.json(
      { error: error?.message || '서버 오류가 발생했습니다.' },
      { status: 500 }
    );
  }
}
