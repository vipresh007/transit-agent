/* Toronto Transit Agent — front end.
 *
 * Deliberately thin. Every judgement about an answer — verified or not, which
 * colour a 504 is, whether grounding is good enough — is made in Python and
 * arrives in the JSON. This file positions things; it does not decide them.
 * Two front ends disagreeing about whether an itinerary is trustworthy would
 * be far worse than either of them looking plain. */

const $ = (id) => document.getElementById(id);

const FRIENDLY = {
  recall_preferences: "checking what you've told me before",
  geocode: "finding the place",
  get_weather: "checking the forecast",
  find_pois: "looking for places nearby",
  find_nearby_stops: "finding nearby stops",
  check_mode_feasibility: "can we get there without buses?",
  plan_journey: "searching the timetable",
  find_direct_trips: "checking departures",
  query_transit: "querying the schedule",
  describe_transit_schema: "reading the schedule layout",
  search_guides: "searching the travel guides",
  remember: "saving a preference",
  forget_preference: "forgetting a preference",
};

let map = null;
let layer = null;
let timer = null;

/* ---- chrome ---------------------------------------------------------- */

async function health() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("health").innerHTML =
      pill("transit.db", s.transit_db) +
      pill("guides.db", s.guides_db) +
      (Object.keys(s.preferences || {}).length
        ? pill(Object.entries(s.preferences).map(([k, v]) => `${k}=${v}`).join(" · "), true)
        : "");
  } catch { /* the page is still usable without it */ }
}

const pill = (text, on) =>
  `<span class="pill${on ? "" : " off"}">${esc(text)}</span>`;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---- live run -------------------------------------------------------- */

function step(html, cls = "") {
  const li = document.createElement("li");
  li.className = cls;
  li.innerHTML = html;
  $("steps").appendChild(li);
  return li;
}

async function ask(question) {
  $("result").hidden = true;
  $("console").hidden = false;
  $("steps").innerHTML = "";
  $("console-title").textContent = "Researching";
  $("go").disabled = true;

  const started = performance.now();
  timer = setInterval(() => {
    $("elapsed").textContent = ((performance.now() - started) / 1000).toFixed(1) + "s";
  }, 100);

  let running = null;
  const { id } = await (await fetch("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, geocode: true }),
  })).json();

  const source = new EventSource(`/api/stream/${id}`);

  source.onmessage = (message) => {
    const e = JSON.parse(message.data);

    if (e.kind === "prefs" && e.remembered?.length) {
      step(`<span class="tick">◆</span> applying ${esc(e.describe)}`, "note");
    } else if (e.kind === "tool_start") {
      running = step(`<span class="tick">◜</span> ${esc(FRIENDLY[e.tool] || e.tool)}…`, "run");
    } else if (e.kind === "tool_call") {
      // Reuse the "running" row rather than adding a second one, so the list
      // reads as progress rather than as an ever-growing log.
      const row = running || step("", "");
      running = null;
      row.className = e.barren ? "warn" : "done";
      row.innerHTML =
        `<span class="tick">${e.barren ? "!" : "✓"}</span> ` +
        `${esc(FRIENDLY[e.tool] || e.tool)}` +
        `<span class="ms">${(e.seconds ?? 0).toFixed(1)}s</span>`;
    } else if (e.kind === "final") {
      $("console-title").textContent = "Building the itinerary";
    } else if (e.kind === "done") {
      render(e.result);
      $("usage").textContent = e.usage || "";
    } else if (e.kind === "error") {
      $("result").hidden = false;
      $("result").innerHTML =
        `<div class="banner"><strong>The run failed.</strong><br>${esc(e.error)}</div>`;
    } else if (e.kind === "finished") {
      source.close();
      clearInterval(timer);
      $("console").hidden = true;
      $("go").disabled = false;
      loadTraces();
    }
  };

  source.onerror = () => {
    source.close();
    clearInterval(timer);
    $("go").disabled = false;
    $("console-title").textContent = "Connection lost";
  };
}

/* ---- rendering ------------------------------------------------------- */

function render(data) {
  const box = $("result");
  box.hidden = false;
  // Kill any poller from the previous answer before the DOM it wrote into is
  // replaced. Otherwise every question leaves a timer behind, fetching for a
  // map that no longer exists.
  stopLive();

  if (data.error) {
    box.innerHTML = `<div class="banner"><strong>Couldn't structure an answer.</strong><br>${esc(data.error)}</div>`;
    return;
  }

  let html = `<p class="summary">${esc(data.summary)}</p>`;
  html += stats(data);

  if (!data.feasible) {
    html += `<div class="banner"><strong>No route found.</strong><br>${esc(data.infeasible_reason)}</div>`;
    box.innerHTML = html + extras(data);
    return;
  }

  html += `<div class="split"><div>${timeline(data)}</div><div><div id="map"></div>
    <div class="tally" id="tally"></div>
    <div class="tally" id="live"></div>${
    data.map.unresolved.length
      ? `<div class="note">Not on the map: ${esc(data.map.unresolved.join(", "))}. ` +
        `Neighbourhood names aren't in the GTFS feed — stop names are intersections.</div>`
      : ""
  }</div></div>`;

  box.innerHTML = html + extras(data);
  drawMap(data);
}

const stats = (d) =>
  `<div class="stats">` +
  Object.entries(d.stats).map(([k, v]) =>
    `<div class="stat ${d.tone[k] || ""}"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`
  ).join("") +
  `<div class="stat"><div class="k">Journey</div><div class="v">${d.total_min} min</div></div>` +
  `</div>`;

function timeline(d) {
  let out = '<div class="tl">';
  for (const leg of d.legs) {
    const walk = leg.mode === "walk";
    out += `<div class="leg ${walk ? "walk" : ""}" style="--c:${esc(leg.colour)}">
      <div class="card">
        <div class="time">${esc(leg.depart)}</div>
        <div class="body">
          <span class="route">${walk ? "WALK" : esc(leg.route + " · " + leg.mode)}</span>
          <div class="where">${esc(leg.origin)} → ${esc(leg.destination)}</div>
        </div>
        <div class="dur">${leg.minutes}m</div>
      </div>
      ${leg.warning ? `<div class="gap">⚠ ${esc(leg.warning)}</div>` : ""}
    </div>`;
  }
  const last = d.legs[d.legs.length - 1];
  if (last) {
    out += `<div class="leg" style="--c:#34d399"><div class="card">
      <div class="time">${esc(last.arrive)}</div>
      <div class="body"><span class="route">ARRIVE</span>
      <div class="where">${esc(last.destination)}</div></div>
    </div></div>`;
  }
  return out + "</div>";
}

function extras(d) {
  let out = "";
  if (d.violations.length) {
    out += `<details class="more"><summary>${d.violations.length} constraint violation(s)</summary><ul>` +
      d.violations.map((v) => `<li><b>${esc(v.kind)}</b> — ${esc(v.detail)}<br><span style="color:#5d6775">${esc(v.fix)}</span></li>`).join("") +
      `</ul></details>`;
  }
  if (d.caveats.length) {
    out += `<details class="more"><summary>${d.caveats.length} caveat(s)</summary><ul>` +
      d.caveats.map((c) => `<li>${esc(c)}</li>`).join("") + `</ul></details>`;
  }
  if (d.grounding.unsupported?.length) {
    out += `<details class="more"><summary>${d.grounding.unsupported.length} unsupported specific(s)</summary>` +
      `<ul><li>${d.grounding.unsupported.map(esc).join(", ")}</li></ul>` +
      `<div class="note">Present in the answer, absent from every tool result. Some are false positives.</div></details>`;
  }
  return out;
}

function drawMap(d) {
  const el = document.getElementById("map");
  if (!el) return;

  if (typeof L === "undefined") {
    el.innerHTML = `<div class="mapfail">Leaflet didn't load.<br>
      <span>The page pulls it from unpkg.com — an extension or offline mode
      will block it. Check the console.</span></div>`;
    return;
  }
  if (!d.viewport || !d.map.points.length) {
    el.innerHTML = `<div class="mapfail">No coordinates resolved.<br>
      <span>Stop names are intersections; neighbourhood names aren't in the
      GTFS feed.</span></div>`;
    return;
  }

  // WAIT FOR A REAL SIZE BEFORE STARTING LEAFLET.
  //
  // Leaflet measures its container once, at construction, and positions every
  // tile and layer from that measurement. This container is created by
  // innerHTML a moment earlier, inside a CSS grid whose track widths are not
  // final — so Leaflet read 0px, and drew the whole journey outside the
  // visible box. A single invalidateSize() on the next frame helped and
  // wasn't reliable: fonts, the backdrop-filter and the grid all settle at
  // different times.
  //
  // So: don't guess when layout is done, WAIT until the element reports a
  // width. Same shape as the rest of this project — measure the thing you
  // actually depend on rather than a proxy for it.
  let frames = 0;
  const build = () => {
    if (el.clientWidth === 0 && frames++ < 60) {
      requestAnimationFrame(build);
      return;
    }
    buildMap(el, d);
  };
  requestAnimationFrame(build);
}

function buildMap(el, d) {
  if (map) { try { map.remove(); } catch {} map = null; }

  try {
    map = L.map(el, { zoomControl: true, attributionControl: true })
      .setView([d.viewport.latitude, d.viewport.longitude], d.viewport.zoom);
  } catch (err) {
    el.innerHTML = `<div class="mapfail">The map failed to start.<br>
      <span>${esc(err.message)}</span></div>`;
    return;
  }

  // Voyager by default, not dark_matter. The dark basemap matched the page
  // beautifully and hid the streets, which is the one thing a transit map has
  // to show. A map you can't read is decoration.
  //
  // Both are free, no key. The toggle is remembered per browser.
  const BASEMAPS = {
    streets: {
      url: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
      label: "Dark",
    },
    dark: {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      label: "Streets",
    },
  };
  let style = localStorage.getItem("basemap") || "streets";

  let tiles = L.tileLayer(BASEMAPS[style].url, {
    attribution: "&copy; OpenStreetMap &copy; CARTO", maxZoom: 19,
  }).addTo(map);

  const Toggle = L.Control.extend({
    options: { position: "topright" },
    onAdd() {
      const btn = L.DomUtil.create("button", "basemap-toggle");
      btn.textContent = BASEMAPS[style].label;
      L.DomEvent.disableClickPropagation(btn);
      btn.onclick = () => {
        style = style === "streets" ? "dark" : "streets";
        localStorage.setItem("basemap", style);
        map.removeLayer(tiles);
        tiles = L.tileLayer(BASEMAPS[style].url, {
          attribution: "&copy; OpenStreetMap &copy; CARTO", maxZoom: 19,
        }).addTo(map);
        tiles.bringToBack();
        btn.textContent = BASEMAPS[style].label;
      };
      return btn;
    },
  });
  map.addControl(new Toggle());

  layer = L.layerGroup().addTo(map);

  let drawn = 0;
  for (const p of d.map.paths) {
    const pts = p.path.map(([lon, lat]) => [lat, lon]);
    // Casing: a wide dark line under a narrower coloured one. Without it a
    // red route over a red-brick street map is nearly invisible, and route
    // colours are the whole point of colouring them.
    L.polyline(pts, {
      color: "#0b1018", weight: p.mode === "walk" ? 7 : 10,
      opacity: 0.55, lineCap: "round", lineJoin: "round",
    }).addTo(layer);

    const line = L.polyline(pts, {
      color: `rgb(${p.colour.join(",")})`,
      weight: p.mode === "walk" ? 4 : 6,
      opacity: p.mode === "walk" ? 0.85 : 1,
      dashArray: p.dashed ? "8 10" : undefined,
      lineCap: "round", lineJoin: "round",
    });
    line.addTo(layer).bindPopup(esc(p.label));
    drawn++;
  }

  for (const s of d.map.points) {
    L.circleMarker([s.lat, s.lon], {
      radius: 7, color: "#0b1018", weight: 3,
      fillColor: "#ffffff", fillOpacity: 1,
    }).addTo(layer).bindPopup(esc(s.name));
  }

  // Re-measure whenever the box actually changes size, not just once. The
  // window resizing, the sidebar reflowing and the fonts loading are all
  // events Leaflet won't hear about on its own.
  if (window.ResizeObserver) {
    new ResizeObserver(() => map.invalidateSize()).observe(el);
  }

  const bounds = L.latLngBounds(d.map.points.map((s) => [s.lat, s.lon]));
  requestAnimationFrame(() => {
    map.invalidateSize();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [50, 50] });
  });

  // Say what was drawn. "The map is empty" and "the map drew nothing" are
  // different problems, and from outside they look the same.
  const tally = document.getElementById("tally");
  if (tally) {
    // Say which lines are real track and which are straight-line guesses.
    // An approximate line is fine; an unlabelled one invites the reader to
    // believe the route runs where it doesn't.
    const exact = d.map.paths.filter((p) => p.exact).length;
    const approx = d.map.paths.filter((p) => !p.exact && p.mode !== "walk").length;
    const walks = d.map.paths.filter((p) => p.mode === "walk").length;

    tally.innerHTML =
      `${d.map.points.length} stops · ${drawn} legs` +
      (exact ? ` · ${exact} on real track` : "") +
      (approx ? ` · <span class="warnish">${approx} approximate</span>` : "") +
      (d.map.unresolved.length
        ? ` · <span class="warnish">${d.map.unresolved.length} unplaced</span>` : "") +
      // Say WHY the dashed lines are straight. Otherwise it reads as a bug,
      // and the reader spends attention on a deliberate choice. Transit
      // geometry comes from shapes.txt; GTFS has no pedestrian network at
      // all, so there is nothing to draw a sidewalk from.
      (walks
        ? `<br><span class="dim">Dashed walks are straight lines — GTFS has ` +
          `route geometry but no footpaths.</span>`
        : "");
  }

  startLive(d);
}

/* ---- live vehicles ---------------------------------------------------- */
/* Positions only. The feed also carries per-stop predictions, and we do not
 * use them: its stop ids disagree with our database (59% of them collide by
 * number while naming a different stop), so a delay would be attached to the
 * wrong place and look exactly like a correct one. A dot on a map claims
 * "a vehicle is here"; a time claims "you will catch this". Only one of those
 * survives a 1% join. */

let liveLayer = null;
let liveTimer = null;

function stopLive() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
}

function startLive(d) {
  stopLive();
  const routes = d.live_routes || [];
  const note = document.getElementById("live");
  if (!note) return;

  if (!routes.length) {
    // Say why rather than showing nothing. Every subway-only journey lands
    // here, and silence would read as a broken feature.
    note.innerHTML = `<span class="dim">No live vehicles: TTC's realtime ` +
      `feed covers buses and streetcars, not the subway.</span>`;
    return;
  }

  const tick = async () => {
    let data;
    try {
      const url = "/api/vehicles?routes=" + encodeURIComponent(routes.join("|"));
      data = await (await fetch(url)).json();
    } catch {
      data = { available: false };
    }

    if (liveLayer) { map.removeLayer(liveLayer); liveLayer = null; }

    // "Couldn't reach the feed" and "nothing is running" must not render the
    // same way. The first is our problem, the second is information.
    if (!data.available) {
      note.innerHTML = `<span class="warnish">Live positions unavailable</span> ` +
        `<span class="dim">— the feed didn't answer. The itinerary above is ` +
        `unaffected.</span>`;
      return;
    }

    liveLayer = L.layerGroup().addTo(map);
    for (const [label, vehicles] of Object.entries(data.routes || {})) {
      for (const v of vehicles) {
        L.circleMarker([v.lat, v.lon], {
          radius: 5, weight: 2, color: "#0b1018",
          fillColor: "#34d399", fillOpacity: 0.95, className: "live-dot",
        }).addTo(liveLayer).bindPopup(
          `${esc(label)} — live position` +
          (v.bearing != null ? `<br>heading ${Math.round(v.bearing)}°` : ""));
      }
    }

    note.innerHTML = data.count
      ? `<span class="livedot"></span>${data.count} vehicle` +
        `${data.count === 1 ? "" : "s"} on ${esc(routes.join(", "))} right now ` +
        `<span class="dim">· positions only, not arrival predictions</span>`
      : `<span class="dim">No vehicles running on ` +
        `${esc(routes.join(", "))} right now.</span>`;
  };

  tick();
  // The feed republishes about every 30s. Polling faster shows the same
  // snapshot twice and is rude to a free public service.
  liveTimer = setInterval(tick, 30000);
}

/* ---- saved runs ------------------------------------------------------ */

async function loadTraces() {
  const runs = await (await fetch("/api/traces")).json();
  $("traces").innerHTML = runs.length
    ? runs.map((r) => `<button class="tcard" data-name="${esc(r.name)}">
         <div class="q">${esc(r.question || "(no question recorded)")}</div>
         <div class="meta"><span>${esc(r.when || r.name)}</span><span>${esc(r.model || "")}</span></div>
       </button>`).join("")
    : `<div class="hint">No saved runs yet.</div>`;

  for (const card of document.querySelectorAll(".tcard")) {
    card.onclick = async () => {
      const data = await (await fetch(`/api/replay/${card.dataset.name}?geocode=true`)).json();
      if (data.detail) { alert(data.detail); return; }
      $("q").value = data.question || "";
      render(data);
      $("usage").textContent = "replayed — 0 requests";
      $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    };
  }
}

/* ---- wiring ---------------------------------------------------------- */

$("ask").onsubmit = (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q) ask(q);
};

for (const b of document.querySelectorAll(".suggest button")) {
  b.onclick = () => { $("q").value = b.dataset.q; ask(b.dataset.q); };
}

health();
loadTraces();
