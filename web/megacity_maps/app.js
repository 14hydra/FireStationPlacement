/* Megacity placement Google Maps viewer */

const state = {
  map: null,
  cityId: null,
  layers: {
    kept: [],
    closed: [],
    opened: [],
    demandHeat: null,
    delta: [],
  },
  cities: [],
};

function $(id) {
  return document.getElementById(id);
}

function fmtSec(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const s = Number(v);
  return `${s.toFixed(1)} s (${(s / 60).toFixed(2)} min)`;
}

function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const x = Number(v);
  return x <= 1.0001 ? `${(x * 100).toFixed(1)}%` : `${x.toFixed(1)}%`;
}

function clearLayer(arrOrObj) {
  if (!arrOrObj) return;
  if (Array.isArray(arrOrObj)) {
    arrOrObj.forEach((m) => m.setMap && m.setMap(null));
    arrOrObj.length = 0;
  } else if (arrOrObj.setMap) {
    arrOrObj.setMap(null);
  }
}

function clearAllLayers() {
  clearLayer(state.layers.kept);
  clearLayer(state.layers.closed);
  clearLayer(state.layers.opened);
  clearLayer(state.layers.delta);
  if (state.layers.demandHeat) {
    state.layers.demandHeat.setMap(null);
    state.layers.demandHeat = null;
  }
}

async function fetchJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function marker(map, pos, opts) {
  return new google.maps.Marker({
    map,
    position: pos,
    ...opts,
  });
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

async function initMap() {
  const key = window.GOOGLE_MAPS_API_KEY;
  if (!key || key.startsWith("__")) {
    $("metricsBody").innerHTML =
      "<strong>Missing Google Maps API key.</strong><br/>Run " +
      "<code>PYTHONPATH=src .venv/bin/python scripts/serve_megacity_maps.py</code> " +
      "with <code>GOOGLE_MAPS_API_KEY</code> set.";
    return;
  }
  await loadScript(
    `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&libraries=visualization`
  );

  state.map = new google.maps.Map($("map"), {
    center: { lat: 20, lng: 20 },
    zoom: 3,
    mapTypeControl: true,
    streetViewControl: false,
    fullscreenControl: true,
    styles: [
      { featureType: "poi", stylers: [{ visibility: "off" }] },
      { featureType: "transit", stylers: [{ visibility: "off" }] },
    ],
  });

  state.cities = await fetchJson("data/cities.json");
  const sel = $("citySelect");
  sel.innerHTML = "";
  state.cities.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.name}${c.metrics ? "" : " (pending)"}`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => selectCity(sel.value));

  ["layerKept", "layerClosed", "layerOpened", "layerDemand", "layerDelta"].forEach((id) => {
    $(id).addEventListener("change", applyLayerVisibility);
  });

  const first = state.cities.find((c) => c.metrics) || state.cities[0];
  if (first) {
    sel.value = first.id;
    await selectCity(first.id);
  }
}

function applyLayerVisibility() {
  const show = {
    kept: $("layerKept").checked,
    closed: $("layerClosed").checked,
    opened: $("layerOpened").checked,
    demand: $("layerDemand").checked,
    delta: $("layerDelta").checked,
  };
  state.layers.kept.forEach((m) => m.setMap(show.kept ? state.map : null));
  state.layers.closed.forEach((m) => m.setMap(show.closed ? state.map : null));
  state.layers.opened.forEach((m) => m.setMap(show.opened ? state.map : null));
  state.layers.delta.forEach((m) => m.setMap(show.delta ? state.map : null));
  if (state.layers.demandHeat) {
    state.layers.demandHeat.setMap(show.demand ? state.map : null);
  }
}

function renderMetrics(summary) {
  const m = summary.metrics || {};
  const delta = m.delta_s;
  const deltaCls = delta == null ? "" : delta < 0 ? "delta-neg" : delta > 0 ? "delta-pos" : "";
  $("metricsBody").innerHTML = `
    <div class="metric-row"><span>Stations today</span><span>${m.n_stations ?? "—"}</span></div>
    <div class="metric-row"><span>Opened / closed</span><span>${m.n_opened ?? "—"} / ${m.n_closed ?? "—"}</span></div>
    <div class="metric-row"><span>Mean today</span><span>${fmtSec(m.average_s_today)}</span></div>
    <div class="metric-row"><span>Mean after move-10</span><span>${fmtSec(m.average_s_plan)}</span></div>
    <div class="metric-row"><span>Demand-weighted Δ</span><span class="${deltaCls}">${
      delta == null ? "—" : `${delta >= 0 ? "+" : ""}${Number(delta).toFixed(1)} s`
    }</span></div>
    <div class="metric-row"><span>≥5 s faster (call-weight)</span><span>${
      m.pct_callweight_5s_faster == null ? "—" : `${Number(m.pct_callweight_5s_faster).toFixed(0)}%`
    }</span></div>
    <div class="metric-row"><span>Within 6 min</span><span>${fmtPct(m.within_6min_today)} → ${fmtPct(
      m.within_6min_plan
    )}</span></div>
    <div class="metric-row"><span>Within 10 min</span><span>${fmtPct(m.within_10min_today)} → ${fmtPct(
      m.within_10min_plan
    )}</span></div>
    <p class="muted" style="margin:0.6rem 0 0">${summary.caveat || ""}</p>
  `;
}

function renderRationale(items) {
  if (!items || !items.length) {
    $("rationaleBody").textContent = "No open/close rationale for this city.";
    return;
  }
  $("rationaleBody").innerHTML = items
    .map((r) => {
      const cls = r.role === "opened" ? "why-open" : "why-close";
      const mark = r.role === "opened" ? "▲ open" : "✕ close";
      const share = r.local_share != null ? `${(r.local_share * 100).toFixed(1)}% local calls` : "";
      const d =
        r.local_delta_s != null
          ? `Δ ${r.local_delta_s >= 0 ? "+" : ""}${r.local_delta_s.toFixed(0)} s`
          : "";
      return `<div class="why-item ${cls}"><strong>${mark} · ${r.name}</strong>${share}${
        share && d ? " · " : ""
      }${d}<br/>${r.why || ""}</div>`;
    })
    .join("");
}

function deltaColor(deltaS) {
  if (deltaS == null || Number.isNaN(deltaS)) return "#94a3b8";
  if (deltaS <= -30) return "#1d5bbf";
  if (deltaS <= -5) return "#6ea0e0";
  if (deltaS < 5) return "#cbd5e1";
  if (deltaS < 30) return "#e07060";
  return "#b91c1c";
}

async function selectCity(cityId) {
  state.cityId = cityId;
  clearAllLayers();
  const base = `data/${cityId}`;
  let summary;
  try {
    summary = await fetchJson(`${base}/summary.json`);
  } catch (e) {
    $("metricsBody").textContent = `No results yet for ${cityId}. Run run_megacity_placement.py.`;
    $("rationaleBody").textContent = "—";
    return;
  }

  renderMetrics(summary);
  state.map.setCenter({ lat: summary.center.lat, lng: summary.center.lon });
  state.map.setZoom(11);

  const [kept, closed, opened, demand, rationale] = await Promise.all([
    fetchJson(`${base}/stations_kept.geojson`).catch(() => ({ features: [] })),
    fetchJson(`${base}/stations_closed.geojson`).catch(() => ({ features: [] })),
    fetchJson(`${base}/stations_opened.geojson`).catch(() => ({ features: [] })),
    fetchJson(`${base}/demand_cells.geojson`).catch(() => ({ features: [] })),
    fetchJson(`${base}/move_rationale.json`).catch(() => []),
  ]);
  renderRationale(rationale);

  (kept.features || []).forEach((f) => {
    const [lng, lat] = f.geometry.coordinates;
    state.layers.kept.push(
      marker(state.map, { lat, lng }, {
        title: f.properties.name || "Station",
        icon: {
          path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
          scale: 4,
          fillColor: "#1e293b",
          fillOpacity: 1,
          strokeWeight: 0,
          rotation: 180,
        },
      })
    );
  });

  (closed.features || []).forEach((f) => {
    const [lng, lat] = f.geometry.coordinates;
    state.layers.closed.push(
      marker(state.map, { lat, lng }, {
        title: `Closed: ${f.properties.name || ""}`,
        label: { text: "✕", color: "#b91c1c", fontSize: "14px", fontWeight: "700" },
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 0.1,
          fillOpacity: 0,
          strokeOpacity: 0,
        },
      })
    );
  });

  (opened.features || []).forEach((f) => {
    const [lng, lat] = f.geometry.coordinates;
    state.layers.opened.push(
      marker(state.map, { lat, lng }, {
        title: `Opened: ${f.properties.name || ""}`,
        icon: {
          path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW,
          scale: 5.5,
          fillColor: "#16a34a",
          fillOpacity: 1,
          strokeWeight: 1,
          strokeColor: "#fff",
        },
      })
    );
  });

  const heatData = [];
  (demand.features || []).forEach((f) => {
    const [lng, lat] = f.geometry.coordinates;
    const w = Number(f.properties.weight || 1);
    heatData.push({ location: new google.maps.LatLng(lat, lng), weight: Math.sqrt(Math.max(w, 1)) });

    const delta = f.properties.response_s_delta;
    if (delta != null) {
      state.layers.delta.push(
        new google.maps.Circle({
          map: state.map,
          center: { lat, lng },
          radius: 180,
          fillColor: deltaColor(Number(delta)),
          fillOpacity: 0.55,
          strokeWeight: 0,
          zIndex: 1,
        })
      );
    }
  });

  if (heatData.length && google.maps.visualization) {
    state.layers.demandHeat = new google.maps.visualization.HeatmapLayer({
      data: heatData,
      radius: 28,
      opacity: 0.55,
      map: state.map,
    });
  }

  applyLayerVisibility();
}

initMap().catch((err) => {
  console.error(err);
  $("metricsBody").textContent = String(err);
});
