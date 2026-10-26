/* EconNN dashboard — lógica interactiva */
(function () {
  "use strict";
  const D = window.ECON_DATA;
  const E = D.escenarios;
  const SC = ["alta", "base", "baja"];
  const VARS = ["u", "theta", "w"];
  const VLAB = { u: "u", theta: "θ", w: "w" };
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const fe = (v, d = 2) => Number(v).toExponential(d);
  const ff = (v, d = 3) => Number(v).toFixed(d);
  const sgn = (v, d = 2) => (v > 0 ? "+" : "") + Number(v).toFixed(d);
  const shockFile = (s) => "shock_" + (s > 0 ? "p" : "m") + Math.abs(s).toFixed(1) + ".png";
  const SHOCKS = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3];

  /* ---------------- Captions ---------------- */
  const CAP = {
    coloc: "Puntos de colocación y distribución temporal 70/30",
    pre: "Convergencia del preentrenamiento en SS (1,000 épocas)",
    ws_loss: "Convergencia del warm-start linealizado (2,000 épocas)",
    ws_irf: "Warm-start: IRF inicial de u, θ, w ante shock +1%",
    faseA: "Fase A — residuos PDE, t ∈ [30, 60] (5,000 épocas)",
    faseB: "Fase B — residuos PDE, dominio completo (20,000 épocas)",
    rar: "RAR — 10,000 puntos de alto residuo añadidos",
    faseC: "Fase C — residuos PDE con pool RAR (10,000 épocas)",
    relo: "ReLoBRaLo — evolución de los pesos adaptativos γ",
    oos_etapas: "Validación OOS por etapa",
    drift: "Policy drift sobre la anchor cloud fija",
    sin_shock: "Ausencia de shock: trayectoria impuesta por hard constraints",
    oos_barras: "Residuos PDE fuera de muestra vs umbral 1e-4",
    fir: "FIR EconNN — shock de productividad +1%",
    bvp: "Shock +1%: EconNN vs solve_bvp",
    bvp_poster: "Shock +1%: EconNN vs solve_bvp (versión póster)",
    res_tray: "Residuos PDE sobre la trayectoria del shock +1%",
    beveridge: "Curva de Beveridge — trayectoria representativa",
    hist: "Distribución de |residuo| — test virgen 100k",
    mapa: "Mapa log10|R| en todo el dominio (a, t)",
    hard: "Hard constraints / ausencia de shock (validación externa)",
  };
  const CATS = {
    entrenamiento: ["pre", "ws_loss", "ws_irf", "faseA", "faseB", "faseC", "relo", "oos_etapas", "drift"],
    colocacion: ["coloc", "rar"],
    validacion: ["sin_shock", "oos_barras", "bvp", "bvp_poster", "res_tray", "hist", "mapa", "hard"],
    respuesta: ["fir"],
    robustez: SHOCKS.map((s) => "shock:" + s),
    diagnostico: ["beveridge"],
  };
  const CAT_LAB = { todas: "Todas", entrenamiento: "Entrenamiento", colocacion: "Colocación", validacion: "Validación", respuesta: "Respuesta", robustez: "Robustez ±3%", diagnostico: "Diagnóstico" };
  const img = (sc, key) => "img/" + sc + "/" + key + ".png";

  /* ---------------- Chart.js defaults ---------------- */
  Chart.defaults.color = "#8aa0b8";
  Chart.defaults.font.family = '"Inter","Segoe UI",system-ui,sans-serif';
  Chart.defaults.borderColor = "rgba(0,229,255,.09)";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = "rgba(4,10,22,.95)";
  Chart.defaults.plugins.tooltip.borderColor = "rgba(0,229,255,.5)";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.maintainAspectRatio = false;
  const thresholdPlugin = {
    id: "threshold",
    afterDatasetsDraw(chart, _a, opts) {
      if (opts == null || opts.y == null) return;
      const y = chart.scales.y.getPixelForValue(opts.y);
      const { left, right } = chart.chartArea;
      const ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = opts.color || "#ff2bd6";
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = opts.color || "#ff2bd6";
      ctx.font = "11px Consolas, monospace";
      const txt = opts.label || "", tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(2,4,9,.85)"; ctx.fillRect(left + 4, y - 16, tw + 8, 14);
      ctx.fillStyle = opts.color || "#ff2bd6"; ctx.fillText(txt, left + 8, y - 5);
      ctx.restore();
    },
  };
  Chart.register(thresholdPlugin);
  const charts = {};
  const mk = (id, cfg) => { if (charts[id]) charts[id].destroy(); charts[id] = new Chart(document.getElementById(id), cfg); return charts[id]; };
  const alpha = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };

  /* ---------------- KaTeX ---------------- */
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
  }

  /* ---------------- Neural background ---------------- */
  (function neural() {
    const cv = $("#neural-bg"); if (!cv) return;
    const ctx = cv.getContext("2d");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W, H, nodes = [], edges = [], pulses = [], dpr = Math.min(window.devicePixelRatio || 1, 2), visible = true;
    function build() {
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      nodes = []; edges = [];
      const cols = W < 700 ? 6 : 9, perCol = W < 700 ? 6 : 9;
      for (let c = 0; c < cols; c++) {
        const n = c === 0 ? 2 : c === cols - 1 ? 3 : perCol;
        for (let i = 0; i < n; i++) {
          nodes.push({ c, x: W * (0.06 + 0.88 * c / (cols - 1)) + (Math.random() - .5) * 20, y: H * ((i + 1) / (n + 1)) + (Math.random() - .5) * 25, r: c === 0 || c === cols - 1 ? 4 : 2.4, ph: Math.random() * 6.28 });
        }
      }
      for (const a of nodes) for (const b of nodes) if (b.c === a.c + 1 && Math.random() < 0.55) edges.push([a, b]);
    }
    function spawn() { if (edges.length) pulses.push({ e: edges[(Math.random() * edges.length) | 0], t: 0, v: 0.008 + Math.random() * 0.014, col: Math.random() < .7 ? "0,229,255" : "255,43,214" }); }
    function frame() {
      if (!visible) { requestAnimationFrame(frame); return; }
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 0.6;
      for (const [a, b] of edges) { ctx.strokeStyle = "rgba(0,229,255,0.07)"; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      if (!reduce && pulses.length < 70 && Math.random() < 0.6) spawn();
      pulses = pulses.filter((p) => {
        p.t += p.v; if (p.t >= 1) { p.e[1].ph = 0; return false; }
        const [a, b] = p.e, x = a.x + (b.x - a.x) * p.t, y = a.y + (b.y - a.y) * p.t;
        ctx.strokeStyle = `rgba(${p.col},0.55)`; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(a.x + (b.x - a.x) * Math.max(0, p.t - .12), a.y + (b.y - a.y) * Math.max(0, p.t - .12)); ctx.lineTo(x, y); ctx.stroke();
        ctx.fillStyle = `rgba(${p.col},0.95)`; ctx.beginPath(); ctx.arc(x, y, 1.8, 0, 6.28); ctx.fill();
        return true;
      });
      for (const n of nodes) {
        n.ph += 0.03; const g = 0.35 + 0.35 * Math.abs(Math.sin(n.ph));
        ctx.fillStyle = `rgba(0,229,255,${g})`; ctx.shadowColor = "#00e5ff"; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 6.28); ctx.fill(); ctx.shadowBlur = 0;
      }
      requestAnimationFrame(frame);
    }
    build(); window.addEventListener("resize", build);
    new IntersectionObserver((en) => { visible = en[0].isIntersecting; }).observe(cv);
    requestAnimationFrame(frame);
  })();

  /* ---------------- Counters & reveal ---------------- */
  $$("[data-count]").forEach((el) => {
    const tgt = +el.dataset.count, t0 = performance.now();
    const step = (t) => { const k = Math.min(1, (t - t0) / 1600); el.textContent = Math.round(tgt * (1 - Math.pow(1 - k, 3))).toLocaleString("en-US"); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  });
  const io = new IntersectionObserver((en) => en.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.05 });
  $$(".reveal").forEach((el) => io.observe(el));
  const navIo = new IntersectionObserver((en) => en.forEach((e) => {
    if (e.isIntersecting) $$("#nav-links a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("section.block").forEach((s) => navIo.observe(s));

  /* ---------------- Scenario switches ---------------- */
  const state = { pipe: "alta", loss: "alta", rar: "alta", val: "alta", rob: "alta" };
  const handlers = {};
  function buildSwitch(el, cur, onChange, opts) {
    const items = opts || SC.map((k) => ({ k, lab: E[k].nombre, sc: true }));
    el.innerHTML = items.map((it) => `<button data-sc="${it.sc ? it.k : ""}" data-k="${it.k}" class="${it.sc ? "" : "gen"} ${it.k === cur ? "on" : ""}">${it.lab}</button>`).join("");
    el.addEventListener("click", (ev) => {
      const b = ev.target.closest("button"); if (!b) return;
      $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.k);
    });
  }
  $$(".sc-switch").forEach((el) => {
    const tg = el.dataset.target;
    buildSwitch(el, state[tg], (k) => { state[tg] = k; handlers[tg] && handlers[tg](); });
  });

  /* ---------------- 02 Modelo ---------------- */
  const PARAMS = [
    ["r", "0.0042", "Tasa de interés real", "5 % anual"], ["λ", "0.0300", "Destrucción de empleo", "Shimer (2005)"],
    ["α", "0.500", "Elasticidad del matching", "Blanchard y Diamond (1989)"], ["β", "0.500", "Poder negociador", "Condición de Hosios"],
    ["z", "0.400", "Valor del desempleo", "Shimer (2005)"], ["χ", "0.470", "Eficiencia del matching", "Calibrado al SS"],
    ["c", "0.524", "Costo de vacante", "Calibrado a la JCC"], ["φ_w", "0.100", "Ajuste salarial (base)", "Gertler y Trigari (2009)"],
    ["φₐ", "0.0305", "Reversión de la productividad", "Hagedorn y Manovskii (2008)"], ["σₐ", "0.007", "Desv. estándar del shock", "Hagedorn y Manovskii (2008)"],
  ];
  $("#tbl-params").innerHTML = "<tr><th>Símb.</th><th style='text-align:left'>Descripción</th><th>Valor</th><th style='text-align:left'>Fuente</th></tr>" +
    PARAMS.map((p) => `<tr><td>${p[0]}</td><td class="txt">${p[2]}</td><td>${p[1]}</td><td class="txt muted">${p[3]}</td></tr>`).join("");
  const SSV = [["ū", "Tasa de desempleo", "0.0600"], ["θ̄", "Tensión del mercado", "1.0000"], ["w̄", "Salario real", "0.9619"], ["v̄ = θ̄ū", "Tasa de vacantes", "0.0600"],
    ["f(θ̄)", "Tasa de hallazgo de empleo", "0.47 / mes"], ["1/f(θ̄)", "Duración media del desempleo", "2.13 meses"]];
  $("#tbl-ss").innerHTML = "<tr><th>Variable</th><th style='text-align:left'>Descripción</th><th>Valor</th></tr>" +
    SSV.map((s) => `<tr><td>${s[0]}</td><td class="txt">${s[1]}</td><td>${s[2]}</td></tr>`).join("");

  /* ---------------- 02 Network SVG ---------------- */
  (function network() {
    const svg = $("#net-svg"); const NS = "http://www.w3.org/2000/svg";
    const el = (t, a, p = svg) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
    const defs = el("defs", {});
    const gr = el("linearGradient", { id: "eg", x1: 0, x2: 1 }, defs);
    el("stop", { offset: "0", "stop-color": "#00e5ff", "stop-opacity": ".55" }, gr); el("stop", { offset: "1", "stop-color": "#ff2bd6", "stop-opacity": ".55" }, gr);
    const flt = el("filter", { id: "gl" }, defs); el("feGaussianBlur", { stdDeviation: "2.5", result: "b" }, flt);
    const mg = el("feMerge", {}, flt); el("feMergeNode", { in: "b" }, mg); el("feMergeNode", { in: "SourceGraphic" }, mg);
    const layers = [
      { n: 2, lab: ["ã", "t̃"], title: "Entrada", col: "#00e5ff" },
      ...Array.from({ length: 6 }, (_, i) => ({ n: 9, title: "Oculta " + (i + 1), sub: "128 · tanh", col: "#8b5cf6" })),
      { n: 3, lab: ["N_u", "N_θ", "N_w"], title: "Latentes", col: "#ff2bd6" },
      { n: 3, lab: ["u", "θ", "w"], title: "Salida económica", col: "#39ff88" },
    ];
    const X0 = 60, X1 = 1040, H = 420, top = 60, bot = 370;
    const pos = layers.map((L, li) => {
      const x = X0 + (X1 - X0) * li / (layers.length - 1);
      return Array.from({ length: L.n }, (_, i) => ({ x, y: L.n === 1 ? (top + bot) / 2 : top + (bot - top) * (L.n === 9 ? i / 8 : (i + 1) / (L.n + 1)) }));
    });
    const eg = el("g", {});
    const edgeEls = [];
    for (let li = 0; li < layers.length - 1; li++) {
      for (const a of pos[li]) for (const b of pos[li + 1]) {
        if (li === layers.length - 2 && pos[li].indexOf(a) !== pos[li + 1].indexOf(b)) continue;
        const ln = el("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: "url(#eg)", "stroke-width": li === layers.length - 2 ? 1.6 : 0.5, "stroke-opacity": 0.35 }, eg);
        ln._a = a; ln._b = b; edgeEls.push(ln);
      }
    }
    // pulses
    const pg = el("g", {});
    const pulseN = 26;
    const pulses = Array.from({ length: pulseN }, () => ({ c: el("circle", { r: 2.6, fill: "#00e5ff", filter: "url(#gl)" }, pg), e: null, t: Math.random() }));
    layers.forEach((L, li) => {
      const g = el("g", {});
      pos[li].forEach((p, i) => {
        const isDots = L.n === 9 && i === 4;
        if (isDots) { el("text", { x: p.x, y: p.y + 5, "text-anchor": "middle", fill: "#8aa0b8", "font-size": 16 }, g).textContent = "⋮"; return; }
        const c = el("circle", { cx: p.x, cy: p.y, r: L.n === 9 ? 8 : 15, fill: "#060b16", stroke: L.col, "stroke-width": 2, filter: "url(#gl)", style: "cursor:pointer" }, g);
        c.addEventListener("mouseenter", () => edgeEls.forEach((e) => { const on = e._a === p || e._b === p; e.setAttribute("stroke-opacity", on ? 1 : 0.08); e.setAttribute("stroke-width", on ? 1.6 : 0.5); }));
        c.addEventListener("mouseleave", () => edgeEls.forEach((e) => { e.setAttribute("stroke-opacity", 0.35); e.setAttribute("stroke-width", 0.5); }));
        if (L.lab) el("text", { x: p.x, y: p.y + 4, "text-anchor": "middle", fill: "#fff", "font-size": 11, "font-family": "Consolas,monospace", "pointer-events": "none" }, g).textContent = L.lab[i];
      });
      el("text", { x: pos[li][0].x, y: 28, "text-anchor": "middle", fill: L.col, "font-size": 13, "font-weight": 700 }, g).textContent = L.title;
      if (L.sub) el("text", { x: pos[li][0].x, y: 44, "text-anchor": "middle", fill: "#8aa0b8", "font-size": 10.5, "font-family": "Consolas,monospace" }, g).textContent = L.sub;
    });
    const xc = (pos[7][0].x + pos[8][0].x) / 2;
    el("rect", { x: xc - 44, y: 395, width: 88, height: 22, rx: 6, fill: "rgba(255,43,214,.12)", stroke: "#ff2bd6" });
    el("text", { x: xc, y: 410, "text-anchor": "middle", fill: "#ff9cf0", "font-size": 10.5, "font-family": "Consolas,monospace" }).textContent = "hard constraints";
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    const nonDot = edgeEls;
    function tick() {
      for (const p of pulses) {
        if (!p.e || p.t >= 1) { p.e = nonDot[(Math.random() * nonDot.length) | 0]; p.t = 0; p.c.setAttribute("fill", Math.random() < .7 ? "#00e5ff" : "#ff2bd6"); }
        p.t += 0.02;
        p.c.setAttribute("cx", p.e._a.x + (p.e._b.x - p.e._a.x) * p.t);
        p.c.setAttribute("cy", p.e._a.y + (p.e._b.y - p.e._a.y) * p.t);
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  })();

  /* ---------------- 03 Pipeline ---------------- */
  const STAGES = [
    { id: "coloc", t: "Puntos de colocación", e: "50.000 LHS", imgs: ["coloc"],
      d: "<b>¿Dónde se evalúan las ecuaciones?</b> En 50.000 puntos (a, t) repartidos con Latin Hypercube. Como los mayores ajustes ocurren justo después del shock, el 30 % de los tiempos se toma de una Beta(0.5, 2), que concentra puntos en los primeros meses; el 70 % restante es uniforme.",
      kv: [["Dominio a", "±5σₐ ⇒ [0.966, 1.036]"], ["Dominio t", "0–60 meses"], ["Puntos con t ≤ 10", "29 % (uniforme: 17 %)"], ["Validación fija", "12.000 pts, nunca entrenados"]] },
    { id: "pre", t: "Pre-SS", e: "1.000 ép.", imgs: ["pre"], oos: "pre",
      d: "<b>Punto de partida estable.</b> Antes de aprender la dinámica, la red se lleva al estado estacionario (1, 1, 1) en todo el dominio, para no partir de pesos aleatorios que desvíen la optimización.",
      kv: [["Optimizador", "Adam"], ["Learning rate", "1e-3 → 3e-4"]] },
    { id: "ws", t: "Warm-start", e: "2.000 ép.", imgs: ["ws_irf"], oos: "post_warmstart",
      d: "<b>Darle la dirección correcta a θ.</b> Se log-linealiza el modelo alrededor del estado estacionario y la red imita esa solución de primer orden. Es clave para θ, que solo converge si parte en la trayectoria de silla. Luego, esta guía se retira de la pérdida.",
      kv: [["Optimizador", "Adam"], ["Learning rate", "5e-4 → 1e-4"]], extra: "eig" },
    { id: "A", t: "Fase A", e: "5.000 ép.", imgs: ["faseA"], oos: "post_A",
      d: "<b>Currículum: empezar por lo fácil.</b> Solo se entrena la segunda mitad del horizonte (t ∈ [30, 60]), cerca del equilibrio, donde la dinámica es más estable.",
      kv: [["Dominio", "t ∈ [30, 60]"], ["Learning rate", "1e-3 → 3e-4"]] },
    { id: "B", t: "Fase B", e: "20.000 ép.", imgs: ["faseB"], oos: "post_B",
      d: "<b>Todo el horizonte.</b> La red aprende la transición completa desde t = 0. En esta fase el error cae más de un orden de magnitud.",
      kv: [["Dominio", "t ∈ [0, 60]"], ["Learning rate", "5e-4 → 5e-5"]] },
    { id: "rar", t: "RAR", e: "+10.000 pts", imgs: ["rar"],
      d: "<b>Reforzar donde la red se equivoca.</b> Se evalúan 40.000 puntos nuevos y los 10.000 con mayor residuo se añaden al conjunto de entrenamiento (Residual-Based Adaptive Refinement).",
      kv: [["Candidatos", "40.000"], ["Pool de la fase C", "60.000"]] },
    { id: "C", t: "Fase C", e: "10.000 ép.", imgs: ["faseC"], oos: "post_C",
      d: "<b>Ajuste fino.</b> Learning rate bajo sobre el conjunto reforzado por el RAR, para reducir oscilaciones.",
      kv: [["Learning rate", "5e-5 → 1e-5"]] },
    { id: "relo", t: "ReLoBRaLo", e: "5.000 pasos", imgs: ["relo"], oos: "post_ReLoBRaLo",
      d: "<b>Equilibrar las tres ecuaciones.</b> Relative Loss Balancing with Random Lookback (Bischof y Kraus, 2021) ajusta los pesos γ según qué tan rápido baja cada residuo, para que ninguna ecuación se descuide.",
      kv: [["Learning rate", "1e-4 → 1e-5"]] },
    { id: "lbfgs", t: "L-BFGS", e: "pulido final", imgs: ["drift"], oos: "post_LBFGS",
      d: "<b>Precisión máxima.</b> Optimizador cuasi-Newton sobre 8.192 puntos fijos en float64, con los pesos γ congelados. Se detiene cuando el error OOS deja de mejorar y la solución ya no cambia (policy drift ≈ 0). El modelo final se elige por error OOS + signos económicos correctos, no solo por la pérdida.",
      kv: [["Puntos fijos", "8.192"], ["Iteraciones/bloque", "20"]], extra: "sel" },
  ];  let curStage = "B";
  $("#pipeline").innerHTML = STAGES.map((s, i) => `<button class="stage ${s.id === curStage ? "on" : ""}" data-id="${s.id}"><div class="n">${String(i + 1).padStart(2, "0")}</div><div class="t">${s.t}</div><div class="e">${s.e}</div></button>`).join("");
  $("#pipeline").addEventListener("click", (ev) => {
    const b = ev.target.closest(".stage"); if (!b) return;
    curStage = b.dataset.id; $$(".stage").forEach((x) => x.classList.toggle("on", x === b)); renderStage();
  });
  function renderStage() {
    const s = STAGES.find((x) => x.id === curStage), sc = state.pipe, e = E[sc];
    let oosHtml = "";
    if (s.oos) {
      const v = s.oos === "pre" ? e.pre_oos.prom : e.etapas[s.oos];
      const ok = v < 1e-4, irf = s.oos === "pre" ? null : e.irf_signos[s.oos];
      oosHtml = `<div class="card" style="background:rgba(0,0,0,.35);margin-top:.8rem"><div class="small muted">Error fuera de muestra (MAE de residuos PDE) al terminar · ${e.nombre}</div>
        <div class="metric-big" style="color:${e.color}">${fe(v, 3)}</div>
        <span class="badge ${ok ? "ok" : "warn"}">${ok ? "< 1e-4" : "≥ 1e-4"}</span>
        ${irf === null ? "" : `<span class="badge ${irf ? "ok" : "warn"}">signos económicos ${irf ? "correctos" : "incorrectos"}</span>`}</div>`;
    }
    let extra = "";
    if (s.extra === "eig") extra = `<p class="small" style="margin-top:.6rem">Según la linealización (${e.nombre}), un shock de +1 % en <i>a</i> hace saltar θ en ≈ <b class="mono">${ff(e.elast, 2)} %</b> en el impacto.</p>`;
    if (s.extra === "sel") extra = `<p class="small" style="margin-top:.6rem"><b>${e.nombre}:</b> ${e.seleccion.nota}.</p>`;
    $("#stage-detail").innerHTML = `<div><h3>${s.t} <span class="muted small">· ${s.e}</span></h3><p>${s.d}</p>
      <dl class="kv">${s.kv.map((k) => `<dt>${k[0]}</dt><dd>${k[1]}</dd>`).join("")}</dl>${extra}${oosHtml}</div>
      <div class="grid" style="gap:.7rem">${s.imgs.map((k) => `<figure><img src="${img(sc, k)}" alt="${CAP[k]}" data-cap="${CAP[k]} — ${e.nombre}"><figcaption>${CAP[k]} · <span style="color:${e.color}">${e.nombre}</span></figcaption></figure>`).join("")}</div>`;
  }
  handlers.pipe = renderStage; renderStage();

  /* ---------------- 04 Pérdida ---------------- */
  const ST_KEYS = ["post_warmstart", "post_A", "post_B", "post_C", "post_ReLoBRaLo", "post_LBFGS"];
  const ST_LAB = ["Warm-start", "Fase A", "Fase B", "Fase C", "ReLoBRaLo", "L-BFGS"];
  const logTicks = { callback: (v) => { const l = Math.log10(v); return Math.abs(l - Math.round(l)) < 1e-9 ? "1e" + Math.round(l) : ""; } };
  mk("ch-etapas", {
    type: "line",
    data: { labels: ["Pre-SS", ...ST_LAB], datasets: SC.map((k) => ({
      label: E[k].nombre + " (φw=" + E[k].phi + ")", data: [E[k].pre_oos.prom, ...ST_KEYS.map((s) => E[k].etapas[s])],
      borderColor: E[k].color, backgroundColor: E[k].color, tension: 0.3, borderWidth: 2.5,
      pointStyle: [true, ...ST_KEYS.map((s) => E[k].irf_signos[s])].map((ok) => (ok ? "circle" : "crossRot")),
      pointRadius: [true, ...ST_KEYS.map((s) => E[k].irf_signos[s])].map((ok) => (ok ? 4 : 9)),
    })) },
    options: { scales: { y: { type: "logarithmic", title: { display: true, text: "MAE PDE promedio OOS" }, ticks: logTicks } },
      plugins: { threshold: { y: 1e-4, label: "criterio 1e-4" }, tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + fe(c.parsed.y, 3) } } } },
  });

  const L = D.logs_baja;
  const OFF = { pre: 0, ws: 1000, A: 3000, B: 8000, C: 28000, relo: 38000, lbfgs: 43000 };
  const PH_COL = { pre: "#8aa0b8", ws: "#8b5cf6", A: "#00e5ff", B: "#38bdf8", C: "#ff9f1c", relo: "#ff2bd6", lbfgs: "#39ff88" };
  const PH_LAB = { pre: "Pre-SS", ws: "Warm-start", A: "Fase A", B: "Fase B", C: "Fase C", relo: "ReLoBRaLo", lbfgs: "L-BFGS (bloques)" };
  const lbx = (i) => OFF.lbfgs + i * 200;
  function logDatasets(k) {
    const ds = (ph, pts) => ({ label: PH_LAB[ph], data: pts, borderColor: PH_COL[ph], backgroundColor: PH_COL[ph], showLine: true, pointRadius: 2.5, borderWidth: 2, tension: 0.2 });
    if (k === "loss") return ["pre", "ws", "A", "B", "C"].map((ph) => ds(ph, L[ph].map((r) => ({ x: OFF[ph] + r[0], y: r[1] })))).concat([ds("relo", L.relo.map((r) => ({ x: OFF.relo + r[0], y: r[1] })))]);
    if (k === "oos") return [ds("B", L.B.map((r) => ({ x: OFF.B + r[0], y: r[2] }))), ds("C", L.C.map((r) => ({ x: OFF.C + r[0], y: r[2] }))), ds("relo", L.relo.map((r) => ({ x: OFF.relo + r[0], y: r[2] }))), ds("lbfgs", L.lbfgs.map((r) => ({ x: lbx(r[0]), y: r[1] })))];
    if (k === "drift") return [ds("A", L.A.map((r) => ({ x: OFF.A + r[0], y: r[2] }))), ds("B", L.B.map((r) => ({ x: OFF.B + r[0], y: r[3] }))), ds("C", L.C.map((r) => ({ x: OFF.C + r[0], y: r[3] }))), ds("relo", L.relo.map((r) => ({ x: OFF.relo + r[0], y: r[3] }))), ds("lbfgs", L.lbfgs.map((r) => ({ x: lbx(r[0]), y: r[2] })))];
    return ["γ_u", "γ_θ", "γ_w"].map((nm, j) => ({ label: nm, data: L.relo.map((r) => ({ x: r[0], y: r[4][j] })), borderColor: ["#38bdf8", "#ff9f1c", "#39ff88"][j], backgroundColor: ["#38bdf8", "#ff9f1c", "#39ff88"][j], showLine: true, pointRadius: 3, borderWidth: 2 }));
  }
  function renderLog(k) {
    const isG = k === "gamma";
    mk("ch-log", { type: "scatter", data: { datasets: logDatasets(k) },
      options: { scales: { x: { type: "linear", title: { display: true, text: isG ? "Paso ReLoBRaLo" : "Paso acumulado del currículum" } },
        y: isG ? { title: { display: true, text: "Peso γ" } } : { type: "logarithmic", ticks: logTicks, title: { display: true, text: { loss: "Loss PDE (MSE)", oos: "MAE OOS (12k)", drift: "Drift RMS" }[k] } } },
        plugins: { threshold: k === "oos" ? { y: 1e-4, label: "criterio 1e-4" } : {}, tooltip: { callbacks: { label: (c) => c.dataset.label + " · paso " + c.parsed.x + ": " + (isG ? ff(c.parsed.y, 3) : fe(c.parsed.y, 3)) } } } } });
  }
  $("#log-tabs").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (!b) return; $$("#log-tabs button").forEach((x) => x.classList.toggle("on", x === b)); renderLog(b.dataset.k); });
  renderLog("loss");

  /* ---------------- 05 Resultados ---------------- */
  const SHOCK_M = {
    theta0: { lab: "θ en el impacto (% vs SS)", get: (k) => [E[k].irf.theta0] },
    umin: { lab: "Mínimo de u (% vs SS)", get: (k) => [E[k].irf.umin] },
    wmax: { lab: "Máximo de w (% vs SS)", get: (k) => [E[k].irf.wmax] },
    mes: { lab: "meses", get: (k) => [E[k].irf.mes_umin, E[k].irf.mes_wmax] },
  };
  function renderShock(m) {
    const isMes = m === "mes";
    mk("ch-shock", { type: "bar",
      data: isMes
        ? { labels: ["Mes del mínimo de u", "Mes del máximo de w"], datasets: SC.map((k) => ({ label: E[k].nombre, data: SHOCK_M.mes.get(k), backgroundColor: alpha(E[k].color, .75), borderColor: E[k].color, borderWidth: 2, borderRadius: 6 })) }
        : { labels: SC.map((k) => E[k].nombre + " (φw=" + E[k].phi + ")"), datasets: [{ data: SC.map((k) => SHOCK_M[m].get(k)[0]), backgroundColor: SC.map((k) => alpha(E[k].color, .75)), borderColor: SC.map((k) => E[k].color), borderWidth: 2, borderRadius: 6 }] },
      options: { indexAxis: isMes ? "y" : "x", plugins: { legend: { display: isMes }, tooltip: { callbacks: { label: (c) => (isMes ? c.dataset.label + ": mes " + c.parsed.x : sgn(c.parsed.y, 3) + " %") } } },
        scales: isMes ? { x: { title: { display: true, text: "meses" } } } : { y: { title: { display: true, text: SHOCK_M[m].lab } } } } });
  }
  $("#shock-tabs").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (!b) return; $$("#shock-tabs button").forEach((x) => x.classList.toggle("on", x === b)); renderShock(b.dataset.m); });
  renderShock("theta0");
  const dvb = (k, j) => (D.comp.diff_vs_base[k] ? ff(D.comp.diff_vs_base[k][j], 2) : "—");
  $("#tbl-esc").innerHTML = "<tr><th>Escenario</th><th>θ₀</th><th>u mín (mes)</th><th>w máx (mes)</th><th>w a 60 m</th><th>Δθ vs Base</th><th>Δu vs Base</th></tr>" +
    SC.map((k) => { const i = E[k].irf; return `<tr><td><span class="sc-dot" style="background:${E[k].color}"></span>${E[k].nombre}</td><td>${sgn(i.theta0, 2)}</td><td>${ff(i.umin, 2)} (${i.mes_umin})</td><td>${sgn(i.wmax, 2)} (${i.mes_wmax})</td><td>${sgn(i.w60, 2)}</td><td>${dvb(k, 0)}</td><td>${dvb(k, 1)}</td></tr>`; }).join("");
  {
    const a = E.alta.irf, b = E.base.irf, c = E.baja.irf;
    $("#reading").innerHTML = [
      `<b>Las vacantes reaccionan de inmediato</b> (θ es <i>forward-looking</i>): el salto es de ${sgn(a.theta0, 1)} % con alta rigidez, frente a ${sgn(c.theta0, 1)} % con baja rigidez. Con salarios rígidos, la ganancia de productividad \\(a-w\\) queda en las firmas, que entonces abren más vacantes.`,
      `<b>El desempleo responde con rezago</b> (<i>backward-looking</i>): toca su mínimo cerca del mes 3 y cae ${ff(Math.abs(a.umin) / Math.abs(c.umin), 1)} veces más con alta rigidez (${ff(a.umin, 2)} %) que con baja (${ff(c.umin, 2)} %).`,
      `<b>El salario sube menos, más tarde y por más tiempo</b>: con alta rigidez alcanza su máximo (${sgn(a.wmax, 2)} %) en el mes ${a.mes_wmax}, frente al mes ${c.mes_wmax} con baja rigidez, y a los 60 meses sigue más alto (${sgn(a.w60, 2)} % vs ${sgn(c.w60, 2)} %).`,
      `<b>Efecto rebote con alta rigidez:</b> después del mes ~20, θ cae bajo su equilibrio y u lo supera, porque el salario sigue alto cuando la productividad ya se disipó. Es una recuperación más lenta e irregular.`,
    ].map((x) => `<li>${x}</li>`).join("");
    if (window.renderMathInElement) renderMathInElement($("#reading"), { delimiters: [{ left: "\\(", right: "\\)", display: false }], throwOnError: false });
  }
  /* ---------------- 07 Laboratorio (solver BVP por disparo) ---------------- */
  const P = { r: 0.0042, lam: 0.03, al: 0.5, be: 0.5, z: 0.4, chi: 0.47, c: 0.524, ab: 1, pa: 0.0305, T: 60 };
  const SS = (() => {
    const f = (th) => (1 - P.be) * (P.ab - P.z) - ((P.r + P.lam) * P.c / (P.chi * Math.pow(th, P.al - 1)) + P.be * P.c * th);
    let lo = 0.01, hi = 10; for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (f(lo) * f(m) <= 0) hi = m; else lo = m; }
    const th = (lo + hi) / 2; return { th, w: (1 - P.be) * P.z + P.be * (P.ab + P.c * th), u: P.lam / (P.lam + P.chi * Math.sqrt(th)) };
  })();
  const cache = new Map();
  function solveBVP(phiw, delta) {
    const key = phiw.toFixed(4) + "|" + delta.toFixed(4);
    if (cache.has(key)) return cache.get(key);
    const N = 1200, h = P.T / N, x0 = Math.log(1 + delta), lthT = Math.log(SS.th);
    const A = (t) => P.ab * Math.exp(x0 * Math.exp(-P.pa * t));
    const F = (t, u, lth, lw) => {
      const th = Math.exp(Math.max(-30, Math.min(30, lth))), w = Math.exp(lw), a = A(t), wn = (1 - P.be) * P.z + P.be * (a + P.c * th);
      return [P.lam * (1 - u) - P.chi * Math.pow(th, P.al) * u, 1 / (1 - P.al) * ((P.r + P.lam) - (a - w) * P.chi * Math.pow(th, P.al - 1) / P.c), phiw * Math.log(wn / w)];
    };
    const run = (s, keep) => {
      let u = SS.u, l = s, lw = Math.log(SS.w), t = 0; const out = keep ? [[0, u, l, lw]] : null;
      for (let i = 0; i < N; i++) {
        const k1 = F(t, u, l, lw), k2 = F(t + h / 2, u + h / 2 * k1[0], l + h / 2 * k1[1], lw + h / 2 * k1[2]);
        const k3 = F(t + h / 2, u + h / 2 * k2[0], l + h / 2 * k2[1], lw + h / 2 * k2[2]), k4 = F(t + h, u + h * k3[0], l + h * k3[1], lw + h * k3[2]);
        u += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); l += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); lw += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
        t += h; l = Math.max(-30, Math.min(30, l));
        if (keep && (i + 1) % 5 === 0) out.push([t, u, l, lw]);
      }
      return keep ? out : l - lthT;
    };
    let lo = lthT - 2, hi = lthT + 2;
    for (let i = 0; i < 70; i++) { const m = (lo + hi) / 2; if (run(m, false) > 0) hi = m; else lo = m; }
    const raw = run((lo + hi) / 2, true);
    const res = { t: raw.map((r) => r[0]), u: raw.map((r) => 100 * (r[1] / SS.u - 1)), theta: raw.map((r) => 100 * (Math.exp(r[2]) / SS.th - 1)), w: raw.map((r) => 100 * (Math.exp(r[3]) / SS.w - 1)), a: raw.map((r) => 100 * (A(r[0]) - 1)) };
    cache.set(key, res); return res;
  }
  const lab = { phi: 0.1, delta: 1, v: "theta", ref: true };
  const VTITLE = { theta: "Tensión laboral θ", u: "Desempleo u", w: "Salario w", a: "Productividad a" };
  function peak(arr, t) { let bi = 1; for (let i = 1; i < arr.length; i++) if (Math.abs(arr[i]) > Math.abs(arr[bi])) bi = i; return [arr[bi], t[bi]]; }
  function renderLab() {
    $("#lab-phi-v").textContent = lab.phi.toFixed(3);
    $("#lab-delta-v").textContent = sgn(lab.delta, 1) + "%";
    const cur = solveBVP(lab.phi, lab.delta / 100);
    const ds = [];
    if (lab.ref && lab.v !== "a") SC.forEach((k) => {
      const r = solveBVP(E[k].phi, lab.delta / 100);
      ds.push({ label: "Ref. " + E[k].nombre + " (φw=" + E[k].phi + ")", data: r.t.map((t, i) => ({ x: t, y: r[lab.v][i] })), borderColor: alpha(E[k].color, .8), borderDash: [6, 4], borderWidth: 1.6, pointRadius: 0, showLine: true });
    });
    ds.unshift({ label: "φw = " + lab.phi.toFixed(3) + " (tu selección)", data: cur.t.map((t, i) => ({ x: t, y: cur[lab.v][i] })), borderColor: "#ffffff", backgroundColor: "#fff", borderWidth: 3, pointRadius: 0, showLine: true });
    if (Math.abs(lab.delta - 1) < 1e-9 && lab.v !== "a") SC.forEach((k) => {
      const i = E[k].irf, pt = lab.v === "theta" ? { x: 0, y: i.theta0 } : lab.v === "u" ? { x: i.mes_umin, y: i.umin } : { x: i.mes_wmax, y: i.wmax };
      ds.push({ label: "EconNN " + E[k].nombre, data: [pt], pointStyle: "rectRot", pointRadius: 9, pointHoverRadius: 11, backgroundColor: E[k].color, borderColor: "#000", borderWidth: 1.5, showLine: false });
    });
    const c = charts["ch-lab"];
    if (c) { c.data.datasets = ds; c.options.scales.y.title.text = VTITLE[lab.v] + " — desv. % vs SS"; c.update("none"); }
    else mk("ch-lab", { type: "scatter", data: { datasets: ds }, options: { animation: false, scales: { x: { type: "linear", min: 0, max: 60, title: { display: true, text: "Meses" } }, y: { title: { display: true, text: VTITLE[lab.v] + " — desv. % vs SS" } } },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } }, tooltip: { callbacks: { label: (c) => c.dataset.label + " · mes " + c.parsed.x.toFixed(2) + ": " + c.parsed.y.toFixed(4) + " %" } } } } });
    const [up, ut] = peak(cur.u, cur.t), [wp, wt] = peak(cur.w, cur.t);
    $("#lab-stats").innerHTML = `<dt>θ impacto</dt><dd>${sgn(cur.theta[0], 3)} %</dd><dt>Pico de u</dt><dd>${sgn(up, 3)} % · mes ${ut.toFixed(2)}</dd>
      <dt>Pico de w</dt><dd>${sgn(wp, 3)} % · mes ${wt.toFixed(2)}</dd><dt>w a 60 meses</dt><dd>${sgn(cur.w[cur.w.length - 1], 3)} %</dd><dt>θ(T) − θ̄</dt><dd>${fe(cur.theta[cur.theta.length - 1], 1)} %</dd>`;
  }
  let labT = 0; const labQ = () => { clearTimeout(labT); labT = setTimeout(renderLab, 16); };
  $("#lab-phi").addEventListener("input", (e) => { lab.phi = +e.target.value; labQ(); });
  $("#lab-delta").addEventListener("input", (e) => { lab.delta = +e.target.value; if (Math.abs(lab.delta) < 0.05) { lab.delta = lab.delta >= 0 ? 0.1 : -0.1; e.target.value = lab.delta; } labQ(); });
  $("#lab-ref").addEventListener("change", (e) => { lab.ref = e.target.checked; labQ(); });
  $$("[data-phi]").forEach((b) => b.addEventListener("click", () => { lab.phi = +b.dataset.phi; $("#lab-phi").value = lab.phi; labQ(); }));
  $("#lab-tabs").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (!b) return; $$("#lab-tabs button").forEach((x) => x.classList.toggle("on", x === b)); lab.v = b.dataset.v; labQ(); });
  renderLab();

  /* ---------------- 07 Validación ---------------- */
  mk("ch-test", { type: "bar", data: { labels: ["Ecuación u", "Ecuación θ", "Ecuación w", "Promedio"], datasets: SC.map((k) => { const t = E[k].test; return { label: E[k].nombre, data: [t.mae_test_u, t.mae_test_theta, t.mae_test_w, t.mae_test_promedio], backgroundColor: alpha(E[k].color, .75), borderColor: E[k].color, borderWidth: 2, borderRadius: 5 }; }) },
    options: { scales: { y: { type: "logarithmic", min: 1e-6, max: 1e-3, ticks: logTicks, title: { display: true, text: "MAE |residuo| en 100.000 pts" } } },
      plugins: { threshold: { y: 1e-4, label: "criterio 1e-4" }, tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + fe(c.parsed.y, 3) } } } } });
  const heat = (v) => { const r = Math.min(1, Math.max(0, (Math.log10(v) + 6) / 3)); return `background:rgba(${Math.round(255 * r)},${Math.round(229 * (1 - r) + 43 * r)},${Math.round(255 * (1 - r) + 214 * r)},${0.12 + 0.35 * r})`; };
  const REG_LAB = { "t 0-10": "meses 0–10", "t 10-30": "meses 10–30", "t 30-60": "meses 30–60", "a baja": "a baja", "a central": "a central", "a alta": "a alta" };
  function renderVal() {
    const sc = state.val, e = E[sc], t = e.test, okAvg = t.mae_test_promedio < 1e-4;
    const maxBvp = Math.max(...VARS.map((v) => e.bvp_err[v][2]));
    const maxRob = Math.max(...e.shocks.map((r) => r.max));
    $("#val-kpis").innerHTML = [
      ["Error PDE en test independiente", fe(t.mae_test_promedio, 2), `<span class="badge ${okAvg ? "ok" : "warn"}">${okAvg ? "cumple < 1e-4" : "levemente sobre 1e-4"}</span>`],
      ["Test / validación", ff(t.ratio_test_validacion, 3), `<span class="badge ok">≈ 1 ⇒ sin sobreajuste</span>`],
      ["Desvío máx. vs solve_bvp (+1 %)", ff(maxBvp, 2) + " pp", `<span class="small muted">sobre respuestas de hasta ${ff(e.irf.theta0, 1)} %</span>`],
      ["Desvío máx. en shocks ±3 %", ff(maxRob, 2) + " pp", `<span class="small muted">12 shocks fuera de selección</span>`],
    ].map((k) => `<div class="card"><div class="small muted">${k[0]}</div><div class="metric-big" style="color:${e.color};font-size:1.6rem">${k[1]}</div>${k[2]}</div>`).join("");
    $("#reg-sc").innerHTML = `· <span style="color:${e.color}">${e.nombre}</span>`;
    $("#tbl-reg").innerHTML = "<tr><th>Región</th><th>Ec. u</th><th>Ec. θ</th><th>Ec. w</th><th>Prom.</th></tr>" +
      e.regiones.map((r) => `<tr><td>${REG_LAB[r.region] || r.region}</td>${r.mae.map((v) => `<td style="${heat(v)}">${fe(v, 1)}</td>`).join("")}<td style="${heat(r.mae_prom)}"><b>${fe(r.mae_prom, 1)}</b></td></tr>`).join("");
    $("#tbl-bvp").innerHTML = "<tr><th>Variable</th><th>RMSE (pp)</th><th>Máx (pp)</th></tr>" + VARS.map((v) => `<tr><td>${VLAB[v]}</td><td>${ff(e.bvp_err[v][0], 3)}</td><td>${ff(e.bvp_err[v][2], 3)}</td></tr>`).join("");
    const setI = (id, k) => { const el = $(id); el.src = img(sc, k); el.dataset.cap = CAP[k] + " — " + e.nombre; };
    setI("#img-bvp", sc === "base" ? "bvp_poster" : "bvp"); setI("#img-mapa", "mapa"); setI("#img-bev", "beveridge");
    renderRobImg();
  }

  const rob = { v: "u", i: 7 };
  function renderRobChart() {
    mk("ch-rob", { type: "line", data: { labels: SHOCKS.map((s) => sgn(s, 1) + "%"), datasets: SC.map((k) => ({ label: E[k].nombre, data: SHOCKS.map((s) => (E[k].shocks.find((r) => r.shock === s && r.var === rob.v) || {}).rmse), borderColor: E[k].color, backgroundColor: E[k].color, borderWidth: 2.5, tension: 0.3, pointRadius: 3 })) },
      options: { scales: { y: { beginAtZero: true, title: { display: true, text: "RMSE de " + VLAB[rob.v] + " vs solve_bvp (pp)" } }, x: { title: { display: true, text: "Tamaño del shock" } } },
        plugins: { tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + ff(c.parsed.y, 4) + " pp" } } } } });
  }
  function renderRobImg() {
    const s = SHOCKS[rob.i], sc = state.val;
    $("#rob-shock-v").textContent = sgn(s, 1) + "%";
    const el = $("#rob-img"); el.src = img(sc, shockFile(s).replace(".png", "")); el.dataset.cap = "Shock " + sgn(s, 1) + "% — EconNN vs solve_bvp — " + E[sc].nombre;
    $("#rob-cap").innerHTML = `<span style="color:${E[sc].color}">${E[sc].nombre}</span> · shock ${sgn(s, 1)} %`;
  }
  $("#rob-var").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (!b) return; $$("#rob-var button").forEach((x) => x.classList.toggle("on", x === b)); rob.v = b.dataset.v; renderRobChart(); });
  $("#rob-shock").addEventListener("input", (e) => { rob.i = +e.target.value; renderRobImg(); });
  renderRobChart();
  handlers.val = renderVal; renderVal();

  mk("ch-diag", { type: "bar", data: { labels: ["σw/σa", "Beveridge anti-horaria", "AC(1) u", "AC(1) θ"], datasets: SC.map((k) => { const g = E[k].diag; return { label: E[k].nombre, data: [g.sigma_ratio, g.beveridge, g.ac_u, g.ac_theta], backgroundColor: alpha(E[k].color, .75), borderColor: E[k].color, borderWidth: 2, borderRadius: 5 }; }) },
    options: { scales: { y: { min: 0.6, max: 1.02 } }, plugins: { tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + ff(c.parsed.y, 3) } } } } });

  /* ---------------- 08 Conclusiones ---------------- */
  const a = E.alta, c = E.baja;
  $("#concl-econ").innerHTML = [
    `<b>La evidencia simulada respalda la hipótesis.</b> Ante el mismo shock, una mayor rigidez salarial amplifica la respuesta de las vacantes y del desempleo, y alarga el ajuste del salario.`,
    `La rigidez actúa a través del <b>margen de la empresa</b>: si el salario no sigue a la productividad, cambia la rentabilidad de abrir vacantes. Es el mecanismo propuesto para resolver el rompecabezas de Shimer.`,
    `Con alta rigidez, la recuperación <b>no es monótona</b>: el desempleo termina sobre su equilibrio cuando el shock se disipa y el salario sigue alto. Esto es coherente con una salida lenta de una crisis.`,
    `<b>La EconNN resuelve el modelo con precisión</b>: cumple las ecuaciones con errores del orden de 1e-5, no se sobreajusta (test ≈ validación), reproduce a <code>solve_bvp</code> y generaliza a shocks no vistos.`,
  ].map((x) => `<li>${x}</li>`).join("");
  $("#concl-next").innerHTML = [
    `Estos resultados corresponden a la <b>etapa 1</b>: calibración de referencia para una economía OCDE.`,
    `<b>Etapa 2:</b> recalibrar con datos oficiales de Chile 2017–2026 (ENE-INE, Bolsa Nacional de Empleo, BDE del Banco Central), separando prepandemia, pandemia y pospandemia.`,
    `En el escenario de baja rigidez el error PDE (${fe(c.test.mae_test_promedio, 2)}) queda apenas sobre el criterio de 1e-4 y se concentra en la ecuación salarial al final del horizonte.`,
    `Alcance: mercado laboral <b>formal</b> chileno. No captura informalidad ni diferencias sectoriales o regionales.`,
    `Aporte: evidencia interpretable para diseñar políticas que reduzcan las fricciones e incentiven la contratación.`,
  ].map((x) => `<li>${x}</li>`).join("");
  $("#tbl-sha").innerHTML = "<tr><th>Escenario</th><th style='text-align:left'>SHA-256 (idéntico antes y después de validar)</th></tr>" +
    SC.map((k) => `<tr><td><span class="sc-dot" style="background:${E[k].color}"></span>${E[k].nombre}</td><td style="text-align:left;white-space:normal" class="hash">${E[k].sha}</td></tr>`).join("");
  /* ---------------- 12 Galería ---------------- */
  const gal = { sc: "todos", cat: "todas" };
  function galItems() {
    const out = [];
    const scs = gal.sc === "todos" ? SC : [gal.sc];
    const cats = gal.cat === "todas" ? Object.keys(CATS) : [gal.cat];
    if ((gal.sc === "todos") && (gal.cat === "todas" || gal.cat === "respuesta")) out.push({ src: "img/comp/comparacion_3_escenarios.png", cap: "Comparación de los 3 escenarios — shock +1%", sc: null });
    for (const cat of cats) for (const sc of scs) for (const k of CATS[cat]) {
      if (k === "bvp_poster" && sc !== "base") continue;
      if (k === "coloc" && sc !== "base" && gal.sc === "todos") continue;
      if (k.startsWith("shock:")) { const s = +k.split(":")[1]; out.push({ src: img(sc, shockFile(s).replace(".png", "")), cap: "Robustez: shock " + sgn(s, 1) + "% vs solve_bvp", sc }); }
      else out.push({ src: img(sc, k), cap: CAP[k], sc });
    }
    return out;
  }
  function renderGal() {
    const items = galItems();
    $("#gal").innerHTML = items.map((it) => `<figure><img loading="lazy" src="${it.src}" alt="${it.cap}" data-cap="${it.cap}${it.sc ? " — " + E[it.sc].nombre : ""}"><figcaption><b>${it.cap}</b>${it.sc ? ` · <span style="color:${E[it.sc].color}">${E[it.sc].nombre}</span>` : ""}</figcaption></figure>`).join("") || "<p class='muted'>Sin figuras para este filtro.</p>";
  }
  buildSwitch($("#gal-sc"), "todos", (k) => { gal.sc = k; renderGal(); }, [{ k: "todos", lab: "Todos" }].concat(SC.map((k) => ({ k, lab: E[k].nombre, sc: true }))));
  buildSwitch($("#gal-cat"), "todas", (k) => { gal.cat = k; renderGal(); }, Object.keys(CAT_LAB).map((k) => ({ k, lab: CAT_LAB[k] })));
  renderGal();

  /* ---------------- Lightbox ---------------- */
  const lb = $("#lb"), lbImg = $("#lb img"), lbCap = $("#lb .cap");
  let lbList = [], lbI = 0;
  function lbShow() { const el = lbList[lbI]; lbImg.src = el.currentSrc || el.src; lbCap.textContent = (el.dataset.cap || el.alt || "") + `  (${lbI + 1}/${lbList.length})`; }
  document.addEventListener("click", (ev) => {
    const im = ev.target.closest("figure img"); if (!im || im.closest("#lb")) return;
    const scope = im.closest(".gal") || document;
    lbList = $$("figure img", scope).filter((x) => x.offsetParent !== null && !x.closest("#lb"));
    lbI = Math.max(0, lbList.indexOf(im)); lbShow(); lb.classList.add("open"); document.body.style.overflow = "hidden";
  });
  const lbClose = () => { lb.classList.remove("open"); document.body.style.overflow = ""; };
  const lbMove = (d) => { lbI = (lbI + d + lbList.length) % lbList.length; lbShow(); };
  $("#lb .x").addEventListener("click", lbClose);
  $("#lb .prev").addEventListener("click", () => lbMove(-1));
  $("#lb .next").addEventListener("click", () => lbMove(1));
  lb.addEventListener("click", (e) => { if (e.target === lb) lbClose(); });
  document.addEventListener("keydown", (e) => { if (!lb.classList.contains("open")) return; if (e.key === "Escape") lbClose(); if (e.key === "ArrowLeft") lbMove(-1); if (e.key === "ArrowRight") lbMove(1); });
  let tx = null; lb.addEventListener("touchstart", (e) => { tx = e.touches[0].clientX; }, { passive: true });
  lb.addEventListener("touchend", (e) => { if (tx === null) return; const dx = e.changedTouches[0].clientX - tx; if (Math.abs(dx) > 60) lbMove(dx < 0 ? 1 : -1); tx = null; });
})();
