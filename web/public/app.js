let matchId = null;
let es = null;
let playerName = "";

const $ = (id) => document.getElementById(id);

const screenSetup = $("screenSetup");
const screenGame  = $("screenGame");
const screenEnd   = $("screenEnd");

const logEl = $("log");
const handLogEl = $("handLog");
const resultsLogEl = $("resultsLog");

const statusLine = $("statusLine");

const handTitleEl = $("handTitle");
const boardEl = $("board");
const streetEl = $("street");
const potLineEl = $("potLine");
const betLineEl = $("betLine");
const oddsLineEl = $("oddsLine");
const toActLineEl = $("toActLine");
const stacksLineEl = $("stacksLine");
const actionsEl = $("actions");
const yourStrengthEl = $("yourStrength");

const heroCardsEl = $("heroCards");
const villCardsEl = $("villCards");
const heroStackLineEl = $("heroStackLine");
const villStackLineEl = $("villStackLine");
const heroPosEl = $("heroPos");
const villPosEl = $("villPos");
const heroBtn = $("heroBtn");
const villBtn = $("villBtn");

const resultBanner = $("resultBanner");
const nextHandBtn = $("nextHandBtn");

const finalLineEl = $("finalLine");
const summaryEl = $("summary");

const ACT = {
  0: "FOLD",
  1: "CHECK",
  2: "CALL",
  3: "BET_HALF_POT",
  4: "RAISE_3X",
  5: "ALLIN",
};

function showScreen(which) {
  screenSetup.classList.add("hidden");
  screenGame.classList.add("hidden");
  screenEnd.classList.add("hidden");
  which.classList.remove("hidden");
}

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function handLog(msg) {
  handLogEl.textContent += msg + "\n";
  handLogEl.scrollTop = handLogEl.scrollHeight;
}

function setBanner(text, kind) {
  resultBanner.textContent = text;
  resultBanner.classList.remove("hidden", "win", "lose");
  resultBanner.classList.add(kind);
}

function clearBanner() {
  resultBanner.classList.add("hidden");
  resultBanner.classList.remove("win", "lose");
  resultBanner.textContent = "";
}

function fmtChipsBb(chips) {
  const bb = (chips / 100.0).toFixed(1);
  return `${chips} (${bb}bb)`;
}

function setDealerButton(heroIsIP) {
  // only ONE D visible, ever
  heroBtn.classList.toggle("hidden", !heroIsIP);
  villBtn.classList.toggle("hidden", heroIsIP);

  heroPosEl.textContent = heroIsIP ? "HERO (IP)" : "HERO (OOP)";
  villPosEl.textContent = heroIsIP ? "VILLAIN (OOP)" : "VILLAIN (IP)";
}

function renderActions(legalActs, enabled) {
  actionsEl.innerHTML = "";
  for (const a of legalActs || []) {
    const btn = document.createElement("button");
    btn.textContent = ACT[a] ?? String(a);
    btn.disabled = !enabled;
    btn.onclick = async () => {
      // disable to prevent double-click spew
      renderActions(legalActs, false);
      await fetch(`/api/match/${matchId}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ act: a, iters: Number($("iters").value || 800) }),
      });
    };
    actionsEl.appendChild(btn);
  }
}

function safeText(el, s) {
  el.textContent = (s ?? "—");
}

function resetGameUI() {
  safeText(handTitleEl, "—");
  safeText(boardEl, "—");
  safeText(streetEl, "—");
  safeText(potLineEl, "—");
  safeText(betLineEl, "Facing bet: —");
  safeText(oddsLineEl, "Pot odds: —");
  safeText(toActLineEl, "—");
  safeText(stacksLineEl, "—");

  safeText(heroCardsEl, "?? ??");
  safeText(villCardsEl, "?? ??");
  safeText(heroStackLineEl, "Stack: —");
  safeText(villStackLineEl, "Stack: —");
  safeText(yourStrengthEl, "—");

  heroBtn.classList.add("hidden");
  villBtn.classList.add("hidden");

  handLogEl.textContent = "";
  clearBanner();
  nextHandBtn.disabled = true;
  actionsEl.innerHTML = "";
}

function whoName(p) {
  return p === 0 ? "HERO" : "VILLAIN";
}

function getHumanHand(msg) {
  if (msg?.your_hand) return msg.your_hand;
  if (msg?.human === "hero") return msg.hero_hand;
  if (msg?.human === "villain") return msg.vill_hand;
  return null;
}

// Show ONLY the human's hole cards during the hand
function renderHumanCardsOnly(msg) {
  const humanHand = getHumanHand(msg) || "?? ??";

  if (msg.human === "hero") {
    safeText(heroCardsEl, humanHand);
    safeText(villCardsEl, "?? ??");
  } else {
    safeText(villCardsEl, humanHand);
    safeText(heroCardsEl, "?? ??");
  }
}

$("startBtn").onclick = async () => {
  playerName = ($("playerName").value || "").trim();
  if (!playerName) {
    alert("Player name is required (used for logging).");
    return;
  }

  resetGameUI();
  logEl.textContent = "";
  statusLine.textContent = "Starting…";

  const payload = {
    playerName,
    hands: Number($("hands").value || 10),
    iters: Number($("iters").value || 800),
    seed: Number($("seed").value || 42),
    pot_bb: Number($("pot").value || 6),
    stacks_bb: Number($("stacks").value || 150),
    no_allin: Boolean($("no_allin").checked),
    human: $("human").value,
    start_ip: $("start_ip").value,
  };

  const r = await fetch("/api/match/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const j = await r.json();
  if (!r.ok) {
    alert(j?.error || "Failed to start match");
    return;
  }

  matchId = j.matchId;
  statusLine.textContent = `Match started (${matchId})`;
  showScreen(screenGame);

  es = new EventSource(`/api/match/${matchId}/stream`);
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    onMsg(msg);
  };
  es.onerror = () => {
    log("SSE disconnected.");
  };

  $("downloadLogBtn").disabled = false;
};

$("downloadLogBtn").onclick = () => {
  const name = ($("playerName").value || "").trim();
  if (!name) return;
  window.open(`/api/log/${encodeURIComponent(name)}`, "_blank");
};

$("backBtn").onclick = async () => {
  if (!matchId) return;
  await fetch(`/api/match/${matchId}/end`, { method: "POST" });
  es?.close();
  es = null;
  matchId = null;
  showScreen(screenEnd);
};

$("quitBtn").onclick = () => {
  es?.close();
  es = null;
  matchId = null;
  showScreen(screenSetup);
};

nextHandBtn.onclick = async () => {
  nextHandBtn.disabled = true;
  clearBanner();
  handLogEl.textContent = "";

  // hide both until we receive the next hand_start
  safeText(heroCardsEl, "?? ??");
  safeText(villCardsEl, "?? ??");

  await fetch(`/api/match/${matchId}/next`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ iters: Number($("iters").value || 800) }),
  });
};

function onMsg(msg) {
  if (msg.type === "match_started") {
    log(`Match started. You are ${msg.human} | Button starts on ${msg.start_ip}`);
    return;
  }

  if (msg.type === "hand_start") {
    clearBanner();
    nextHandBtn.disabled = true;
    actionsEl.innerHTML = "";

    safeText(handTitleEl, `Hand ${msg.hand_num}`);
    handLogEl.textContent = "";

    // only show human hole cards
    renderHumanCardsOnly(msg);

    safeText(yourStrengthEl, msg.your_strength || "—");
    setDealerButton(Boolean(msg.hero_is_ip));

    heroStackLineEl.textContent = `Stack: ${fmtChipsBb(msg.hero_stack)}`;
    villStackLineEl.textContent = `Stack: ${fmtChipsBb(msg.vill_stack)}`;

    handLog(`=== Hand ${msg.hand_num} ===`);
    handLog(`Button: ${msg.hero_is_ip ? "HERO" : "VILLAIN"}`);
    return;
  }

  if (msg.type === "state") {
    safeText(boardEl, msg.board || "—");
    safeText(streetEl, msg.street_name || "—");
    potLineEl.textContent = `Pot: ${fmtChipsBb(msg.pot)}`;

    heroStackLineEl.textContent = `Stack: ${fmtChipsBb(msg.hero_stack)}`;
    villStackLineEl.textContent = `Stack: ${fmtChipsBb(msg.vill_stack)}`;
    stacksLineEl.textContent = `H: ${fmtChipsBb(msg.hero_stack)} · V: ${fmtChipsBb(msg.vill_stack)}`;

    const toAct = msg.to_act; // 0/1
    toActLineEl.textContent = whoName(toAct);

    if (msg.street_bet_open) {
      betLineEl.textContent = `Facing bet: ${fmtChipsBb(msg.bet_amount)} by ${whoName(msg.bet_by)}`;
      const toCall = Number(msg.bet_amount || 0);
      const pot = Number(msg.pot || 0);
      const odds = (toCall <= 0) ? 0 : (toCall / (pot + toCall));
      oddsLineEl.textContent = `Pot odds: ${(odds * 100).toFixed(1)}%`;
    } else {
      betLineEl.textContent = "Facing bet: —";
      oddsLineEl.textContent = "Pot odds: —";
    }

    // re-apply every state so opponent never "leaks"
    renderHumanCardsOnly(msg);

    safeText(yourStrengthEl, msg.your_strength || "—");

    const humanIs = (msg.human === "hero") ? 0 : 1;
    const enabled = (toAct === humanIs);

    renderActions(msg.legal_actions || [], enabled);
    return;
  }

  if (msg.type === "action") {
    handLog(`${whoName(msg.actor)} → ${msg.act_name}`);
    return;
  }

  if (msg.type === "hand_end") {
    // reveal both at end
    safeText(heroCardsEl, msg.hero_hand || "?? ??");
    safeText(villCardsEl, msg.vill_hand || "?? ??");
    safeText(boardEl, msg.final_board || "—");

    const youHuman = $("human").value; // "hero" or "villain"
    const youWon =
      (youHuman === "hero" && msg.winner === 0) ||
      (youHuman === "villain" && msg.winner === 1);

    const net = (youHuman === "hero") ? msg.hero_net : msg.vill_net;
    const sign = net >= 0 ? "+" : "";
    setBanner(`${youWon ? "You win" : "You lose"} · net ${sign}${net} chips`, youWon ? "win" : "lose");

    handLog(`---`);
    handLog(`Final board: ${msg.final_board}`);
    handLog(`Winner: ${whoName(msg.winner)}  (ended_by_fold=${msg.ended_by_fold})`);

    renderActions([], false);
    return;
  }

  if (msg.type === "await_next_hand") {
    nextHandBtn.disabled = false;
    handLog(`(paused) click Next hand when ready`);
    return;
  }

  if (msg.type === "match_end") {
    finalLineEl.textContent = msg.reason || "Match finished.";
    summaryEl.textContent = msg.summary || "";
    resultsLogEl.textContent = msg.results || "";
    showScreen(screenEnd);
    return;
  }

  if (msg.type === "error") {
    alert(msg.message || "Worker error");
    return;
  }

  log(`(unknown msg) ${JSON.stringify(msg)}`);
}
