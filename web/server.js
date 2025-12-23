import express from "express";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import crypto from "crypto";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// project root is one level above /web
const ROOT = path.resolve(__dirname, "..");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Serve frontend from ./public
const publicDir = path.join(__dirname, "public");
app.use(express.static(publicDir));

// Data directory for logs
const dataDir = path.join(__dirname, "data");
const logsDir = path.join(dataDir, "logs");
fs.mkdirSync(logsDir, { recursive: true });

// Match registry
const matches = new Map(); // matchId -> { proc, clients:Set(res), playerName, settings, handActions:[], lastHandStart:null }

function sanitizeName(name) {
  return String(name || "")
    .trim()
    .replace(/[^\w\- ]+/g, "")
    .replace(/\s+/g, "_")
    .slice(0, 64);
}

function logPathForPlayer(playerName) {
  const safe = sanitizeName(playerName);
  return path.join(logsDir, `${safe}.jsonl`);
}

function broadcast(matchId, obj) {
  const m = matches.get(matchId);
  if (!m) return;
  const s = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of m.clients) res.write(s);
}

function appendHandLog(playerName, record) {
  const p = logPathForPlayer(playerName);
  fs.appendFileSync(p, JSON.stringify(record) + "\n", "utf8");
}

app.post("/api/match/start", (req, res) => {
  const {
    playerName,
    hands = 10,
    iters = 800,
    seed = 42,
    c = 1.4,
    pot_bb = 6.0,
    stacks_bb = 150.0,
    no_allin = false,
    human = "hero",
    start_ip = "hero",
    rollout_bet_freq = 0.55,
  } = req.body ?? {};

  const safePlayer = sanitizeName(playerName);
  if (!safePlayer) return res.status(400).json({ error: "playerName is required" });

  const id = crypto.randomBytes(8).toString("hex");

  const scriptPath = path.join(ROOT, "scripts", "play_vs_agent_worker.py");

  const py = spawn(
    process.platform === "win32" ? "python" : "python3",
    ["-u", scriptPath],
    {
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONPATH: ROOT, // key fix: lets worker import `poker.*`
      },
      cwd: ROOT,
    }
  );

  const m = {
    proc: py,
    clients: new Set(),
    playerName: safePlayer,
    settings: { hands, iters, seed, c, pot_bb, stacks_bb, no_allin, human, start_ip, rollout_bet_freq },
    handActions: [],
    lastHandStart: null,
  };
  matches.set(id, m);

  // Parse worker stdout line-by-line JSON
  let buf = "";
  py.stdout.on("data", (chunk) => {
    buf += chunk.toString("utf8");
    while (true) {
      const idx = buf.indexOf("\n");
      if (idx < 0) break;
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;

      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        broadcast(id, { type: "error", message: `Bad JSON from worker: ${line}` });
        continue;
      }

      // ---- server-side logging assembly ----
      if (msg.type === "hand_start") {
        m.handActions = [];
        m.lastHandStart = {
          hand_num: msg.hand_num,
          hero_is_ip: msg.hero_is_ip,
          human: msg.human,
          started_at: new Date().toISOString(),
          hero_stack_start: msg.hero_stack,
          vill_stack_start: msg.vill_stack,
          hero_hand: msg.hero_hand,
          vill_hand: msg.vill_hand,
        };
      } else if (msg.type === "action") {
        m.handActions.push({
          actor: msg.actor,
          act: msg.act,
          act_name: msg.act_name,
          t: new Date().toISOString(),
        });
      } else if (msg.type === "hand_end") {
        const rec = {
          player: m.playerName,
          matchId: id,
          settings: m.settings,
          hand: {
            ...(m.lastHandStart || {}),
            final_board: msg.final_board,
            hero_hand: msg.hero_hand,
            vill_hand: msg.vill_hand,
            winner: msg.winner,
            hero_net: msg.hero_net,
            vill_net: msg.vill_net,
            hero_stack_final: msg.hero_stack,
            vill_stack_final: msg.vill_stack,
            ended_by_fold: msg.ended_by_fold,
            actions: m.handActions,
            ended_at: new Date().toISOString(),
          },
        };
        appendHandLog(m.playerName, rec);
      }

      broadcast(id, msg);
    }
  });

  py.stderr.on("data", (chunk) => {
    broadcast(id, { type: "error", message: chunk.toString("utf8") });
  });

  py.on("exit", (code) => {
    broadcast(id, { type: "match_end", reason: `Worker exited (code=${code})` });
    matches.delete(id);
  });

  // Send start_match to worker
  py.stdin.write(
    JSON.stringify({
      type: "start_match",
      seed,
      hands,
      iters,
      c,
      pot_bb,
      stacks_bb,
      no_allin,
      human,
      start_ip,
      rollout_bet_freq,
    }) + "\n"
  );

  res.json({ matchId: id });
});

app.get("/api/match/:id/stream", (req, res) => {
  const id = req.params.id;
  const m = matches.get(id);
  if (!m) return res.status(404).end();

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  m.clients.add(res);
  res.write(`data: ${JSON.stringify({ type: "connected", matchId: id })}\n\n`);

  req.on("close", () => {
    m.clients.delete(res);
  });
});

app.post("/api/match/:id/action", (req, res) => {
  const id = req.params.id;
  const m = matches.get(id);
  if (!m) return res.status(404).json({ error: "unknown matchId" });

  const { act, iters = 800 } = req.body ?? {};
  if (typeof act !== "number") return res.status(400).json({ error: "act must be a number" });

  m.proc.stdin.write(JSON.stringify({ type: "human_action", act, iters }) + "\n");
  res.json({ ok: true });
});

app.post("/api/match/:id/next", (req, res) => {
  const id = req.params.id;
  const m = matches.get(id);
  if (!m) return res.status(404).json({ error: "unknown matchId" });

  const { iters = 800 } = req.body ?? {};
  m.proc.stdin.write(JSON.stringify({ type: "next_hand", iters }) + "\n");
  res.json({ ok: true });
});

app.post("/api/match/:id/end", (req, res) => {
  const id = req.params.id;
  const m = matches.get(id);
  if (!m) return res.status(404).json({ error: "unknown matchId" });

  try {
    m.proc.kill();
  } catch {}
  matches.delete(id);

  res.json({ ok: true });
});

app.get("/api/log/:player", (req, res) => {
  const safe = sanitizeName(req.params.player);
  if (!safe) return res.status(400).send("bad player");

  const p = logPathForPlayer(safe);
  if (!fs.existsSync(p)) {
    res.setHeader("Content-Type", "text/plain");
    return res.status(404).send("No logs yet for that player name.");
  }

  res.setHeader("Content-Type", "application/json");
  res.setHeader("Content-Disposition", `attachment; filename="${safe}.jsonl"`);
  fs.createReadStream(p).pipe(res);
});

app.listen(PORT, () => console.log(`UI: http://localhost:${PORT}`));
