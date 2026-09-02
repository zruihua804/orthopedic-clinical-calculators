import { timingSafeEqual } from 'node:crypto';
import { SCAPHOID_EVIDENCE } from '../knowledge/scaphoid-evidence.js';

const requests = new Map();
const MAX_CHARS = 1200;
const MAX_TURNS = 6;
const MAX_PER_HOUR = Number(process.env.MAX_REQUESTS_PER_HOUR || 20);

function json(res, status, payload) {
  res.status(status).setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  return res.json(payload);
}

function matchesPassword(received, expected) {
  if (!received || !expected) return false;
  const a = Buffer.from(received);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

function allowRequest(req) {
  const ip = (req.headers['x-forwarded-for'] || 'unknown').split(',')[0].trim();
  const now = Date.now();
  const windowStart = now - 60 * 60 * 1000;
  const active = (requests.get(ip) || []).filter((t) => t > windowStart);
  if (active.length >= MAX_PER_HOUR) return false;
  active.push(now);
  requests.set(ip, active);
  return true;
}

const SYSTEM_PROMPT = `你是“腕舟骨知识问答”助手。只回答腕舟骨骨折、隐匿骨折、延迟愈合/不愈合、近端极活性/可重建性、SNAC、相关影像与术后随访的教学性问题。\n\n回答规则：\n1. 优先使用下方“教材要点”，并在回答末尾给出“依据：”和相应教材卡号/页码。\n2. 教材未覆盖的部分，可使用一般医学知识，但必须标注为“通用知识补充”，不得伪造教材出处。\n3. 不给出个体化诊断、处方、手术决定或确定性预后；对具体病例仅说明需要补全的信息、可能的分流逻辑和应咨询手外科的原因。\n4. 不处理非腕舟骨主题、患者可识别信息、紧急医疗指令或用药剂量；简短说明范围并建议适当专业帮助。\n5. 对 AVN/低活性保持审慎：非增强 MRI 单一信号不能确诊，需结合增强 MRI、CT 结构、术中所见与可固定性。\n6. 使用中文，结构清晰、克制，不引用教材长段原文。\n\n教材要点：\n${SCAPHOID_EVIDENCE}`;

export default async function handler(req, res) {
  if (req.method !== 'POST') return json(res, 405, { error: '仅支持 POST 请求。' });
  const { accessPassword, question, history = [] } = req.body || {};
  if (!matchesPassword(accessPassword, process.env.APP_ACCESS_PASSWORD)) return json(res, 401, { error: '访问密码错误或服务端尚未配置。' });
  if (typeof question !== 'string' || !question.trim() || question.length > MAX_CHARS) return json(res, 400, { error: `问题须为 1–${MAX_CHARS} 个字符。` });
  if (!allowRequest(req)) return json(res, 429, { error: '本小时问答次数已达上限，请稍后再试。' });
  if (!process.env.DEEPSEEK_API_KEY) return json(res, 503, { error: '服务端尚未配置 DeepSeek API 密钥。' });

  const safeHistory = Array.isArray(history) ? history.slice(-MAX_TURNS).filter((m) => m && ['user', 'assistant'].includes(m.role) && typeof m.content === 'string' && m.content.length <= 1800) : [];
  try {
    // 本应用需要直接显示简短教学回答。V4-Flash 默认开启思考模式；在较短
    // token 预算下，可能只返回 reasoning 而没有可显示的正文，因此这里明确关闭。
    const model = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash';
    const upstream = await fetch('https://api.deepseek.com/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}` },
      body: JSON.stringify({
        model,
        thinking: { type: 'disabled' },
        temperature: 0.2,
        max_tokens: 900,
        messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...safeHistory, { role: 'user', content: question.trim() }]
      })
    });
    const data = await upstream.json();
    if (!upstream.ok) {
      console.error('DeepSeek request failed', { status: upstream.status, model, type: data?.error?.type });
      return json(res, 502, { error: data?.error?.message || `模型服务返回 ${upstream.status}，请检查 DeepSeek 密钥、额度与模型配置。` });
    }
    const content = data?.choices?.[0]?.message?.content;
    const answer = typeof content === 'string' ? content.trim() : '';
    if (!answer) {
      console.error('DeepSeek response had no displayable content', { model, finishReason: data?.choices?.[0]?.finish_reason });
      return json(res, 502, { error: '模型未产生可显示正文，请稍后重试；如持续出现，请检查 Vercel 日志。' });
    }
    return json(res, 200, { answer, notice: '教学性回答，不替代完整查体、影像判读或手外科会诊。' });
  } catch (error) {
    return json(res, 502, { error: '连接模型服务失败，请稍后重试。' });
  }
}
