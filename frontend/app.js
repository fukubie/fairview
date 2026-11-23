/* Fairview dashboard. Vanilla JS, no dependencies, CSP-safe (no inline
 * script/style, no innerHTML with dynamic strings). Camera frames are sent to
 * the server only; nothing image-related is ever written to browser storage. */
'use strict';
(() => {
  // ------------------------------------------------------------------ //
  // DOM helpers
  // ------------------------------------------------------------------ //
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const PROPS = new Set(['value', 'checked', 'disabled', 'hidden', 'selected', 'readOnly', 'open', 'htmlFor']);
  const SVG_NS = 'http://www.w3.org/2000/svg';

  function applyProps(el, props) {
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') el.setAttribute('class', value);
      else if (key === 'text') el.textContent = value;
      else if (key === 'on') for (const [evt, fn] of Object.entries(value)) el.addEventListener(evt, fn);
      else if (key === 'data') for (const [k, v] of Object.entries(value)) el.dataset[k] = v;
      else if (PROPS.has(key)) el[key] = value;
      else el.setAttribute(key, value === true ? '' : String(value));
    }
  }
  function addKids(el, kids) {
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    applyProps(el, props);
    return addKids(el, kids);
  }
  function s(tag, props, ...kids) {
    const el = document.createElementNS(SVG_NS, tag);
    applyProps(el, props);
    return addKids(el, kids);
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
  function fill(el, ...kids) { return addKids(clear(el), kids); }
  function setText(el, value) { if (el.textContent !== value) el.textContent = value; }

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const motionStyles = getComputedStyle(document.documentElement);
  const standardMotion = parseFloat(motionStyles.getPropertyValue('--motion-standard'));
  const entranceMotion = parseFloat(motionStyles.getPropertyValue('--motion-entrance'));
  const motionEasing = motionStyles.getPropertyValue('--ease').trim();
  const activeMotion = new Set();
  function animate(el, frames, duration = standardMotion, delay = 0) {
    if (reducedMotion.matches || !el.animate || el.closest('[hidden]')) return;
    const effect = el.animate(frames, { duration, delay, easing: motionEasing, fill: 'backwards' });
    activeMotion.add(effect);
    const done = () => activeMotion.delete(effect);
    effect.onfinish = done;
    effect.oncancel = done;
    return effect;
  }
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) activeMotion.forEach((effect) => effect.cancel());
  });

  const entrances = new WeakMap();
  const entranceObserver = window.IntersectionObserver ? new IntersectionObserver((entries) => {
    let stagger = 0;
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entranceObserver.unobserve(entry.target);
      const enter = entrances.get(entry.target);
      entrances.delete(entry.target);
      if (!reducedMotion.matches) enter(stagger++ * 65);
    }
  }, { threshold: 0, rootMargin: '0px 0px -24px' }) : null;
  function onEnter(el, callback) {
    if (!entranceObserver) return;
    entrances.set(el, callback); entranceObserver.observe(el);
  }

  const counted = new Set();
  function countOnEnter(key, el, value, format) {
    if (!Number.isFinite(value) || value === 0 || counted.has(key)) return;
    const finalText = format(value);
    onEnter(el, () => {
      if (counted.has(key) || !el.isConnected || el.textContent !== finalText) return;
      counted.add(key);
      const visual = h('span', { 'aria-hidden': 'true', text: format(0) });
      const fixed = h('span', { class: 'counter__final', text: finalText });
      const count = h('span', { class: 'counter' }, fixed, visual);
      fill(el, count);
      const start = performance.now();
      function frame(now) {
        if (!el.contains(visual)) return;
        const progress = reducedMotion.matches || el.closest('[hidden]') ? 1 : Math.min(1, (now - start) / entranceMotion);
        setText(visual, format(value * (1 - (1 - progress) ** 3)));
        if (progress < 1) requestAnimationFrame(frame);
        else setText(el, finalText);
      }
      requestAnimationFrame(frame);
    });
  }

  const ICONS = {
    check: ['M20 6 9 17l-5-5'],
    cross: ['M18 6 6 18', 'M6 6l12 12'],
    question: ['M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4.5', 'M12 18h.01'],
    alert: ['M12 3 2 20h20L12 3z', 'M12 10v4', 'M12 17h.01'],
    camera: ['M3 8a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z', 'M16 10.5l5-3v9l-5-3'],
    sliders: ['M4 6h10', 'M18 6h2', 'M4 12h4', 'M12 12h8', 'M4 18h12', 'M20 18h0', 'M16 4v4', 'M10 10v4', 'M18 16v4'],
    minus: ['M6 12h12'],
    scan: ['M4 8V6a2 2 0 0 1 2-2h2', 'M16 4h2a2 2 0 0 1 2 2v2', 'M20 16v2a2 2 0 0 1-2 2h-2', 'M8 20H6a2 2 0 0 1-2-2v-2', 'M9 10h.01', 'M15 10h.01', 'M9.5 15a3.5 3.5 0 0 0 5 0'],
    users: ['M16 20v-1.5a3.5 3.5 0 0 0-3.5-3.5h-5A3.5 3.5 0 0 0 4 18.5V20', 'M10 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z', 'M20 20v-1.5a3.5 3.5 0 0 0-2.5-3.35', 'M15.5 4.15a3.5 3.5 0 0 1 0 6.7'],
    list: ['M9 6h11', 'M9 12h11', 'M9 18h11', 'M4.5 6h.01', 'M4.5 12h.01', 'M4.5 18h.01'],
    chart: ['M4 20V10', 'M10 20V4', 'M16 20v-7', 'M22 20H2'],
    columns: ['M4 5a1 1 0 0 1 1-1h5v16H5a1 1 0 0 1-1-1z', 'M14 4h5a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-5z'],
    twin: ['M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6z', 'M16 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6z', 'M3 19a5 5 0 0 1 9 0', 'M12 19a5 5 0 0 1 9 0'],
    settings: ['M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z', 'M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'],
    link: ['M10 14a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1 1', 'M14 10a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1-1'],
    home: ['M4 11 12 4l8 7', 'M6 9.5V20h12V9.5'],
  };
  function icon(name, cls) {
    return s('svg', { class: 'icon' + (cls ? ' ' + cls : ''), viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
      (ICONS[name] || []).map((d) => s('path', { d })));
  }

  // ------------------------------------------------------------------ //
  // Vocabulary
  // ------------------------------------------------------------------ //
  const ZONES = {
    MATCH_CONFIRMED: { label: 'Match confirmed', icon: 'check', box: '#17b26a', badge: '#067647', dash: [] },
    UNCERTAIN_REVIEW: { label: 'Needs review', icon: 'question', box: '#f79009', badge: '#b54708', dash: [8, 5] },
    NO_MATCH: { label: 'No match', icon: 'cross', box: '#d0d5dd', badge: '#344054', dash: [2, 4] },
    SPOOF_DETECTED: { label: 'Possible spoof', icon: 'alert', box: '#f04438', badge: '#b42318', dash: [12, 4, 2, 4] },
  };
  // How the camera overlay draws a confirmed match with a banned person.
  const BANNED_STYLE = { label: 'Banned', icon: 'alert', box: '#f04438', badge: '#b42318', dash: [] };
  const isBanned = (name) => Boolean(name) && S.subjects.some((x) => x.banned && x.subject === name);
  const REASONS = {
    LIVENESS_BELOW_MINIMUM: 'This may be a photo or screen, not a live person',
    LOW_IMAGE_QUALITY: 'The image is too unclear to decide',
    GRAY_ZONE_SIMILARITY: 'Looks similar, but not similar enough to be sure',
    NARROW_CANDIDATE_MARGIN: 'Two enrolled people look too alike to tell apart',
    INVALID_SCORE: 'The face could not be measured reliably',
    SCORE_ABOVE_HIGH_THRESHOLD: 'Strong, unambiguous match',
    SCORE_BELOW_LOW_THRESHOLD: 'Not an enrolled employee. Treat as a visitor',
    EMPTY_GALLERY: 'No employees are enrolled yet',
    KNOWN_VISITOR: 'Already seen this session as a visitor, not an enrolled employee',
    NO_FACE_DETECTED: 'No face detected',
    INCONSISTENT_IDENTITY: 'Looks like a different person from the other photos',
    EMBEDDING_FAILED: 'The face could not be processed',
    MULTIPLE_FACES: 'More than one face in the photo',
    INCOMPLETE_ENROLLMENT: 'This person needs more enrollment photos before they can be confirmed',
  };
  const humanize = (code) => String(code || '').toLowerCase().replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
  const reasonLabel = (code) => REASONS[code] || humanize(code);
  const zoneLabel = (zone) => (ZONES[zone] ? ZONES[zone].label : humanize(zone));

  /* The single place that turns a decision into display text. An identity is
   * returned only when the engine itself asserted one (zone MATCH_CONFIRMED,
   * `subject` set); `top1` is never used as a name here. */
  function verdictOf(decision) {
    if (!decision) return { zone: 'NO_MATCH', title: 'No decision', name: null, reason: '' };
    const zone = ZONES[decision.zone] ? decision.zone : 'UNCERTAIN_REVIEW';
    const confirmed = zone === 'MATCH_CONFIRMED' && typeof decision.subject === 'string' && decision.subject !== '';
    // Someone who is not enrolled gets a number for this session, never a name.
    const visitor = zone === 'NO_MATCH' && Number.isInteger(decision.visitor) ? `Visitor ${decision.visitor}` : null;
    return {
      zone,
      title: confirmed ? decision.subject : (visitor || zoneLabel(zone === 'MATCH_CONFIRMED' ? 'UNCERTAIN_REVIEW' : zone)),
      kicker: confirmed ? (decision.mode === 'verify' ? 'Identity verified' : 'Match confirmed') : (visitor ? 'No match' : ''),
      name: confirmed ? decision.subject : null,
      visitor,
      reason: reasonLabel(decision.reason),
    };
  }

  const num = (v, d = 2) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '–');
  function fmtRate(v) {
    if (typeof v !== 'number' || !isFinite(v)) return 'n/a';
    if (v === 0) return '0%';
    const p = v * 100;
    if (p >= 10) return p.toFixed(1) + '%';
    if (p >= 0.1) return p.toFixed(2) + '%';
    return Number(p.toPrecision(2)).toString() + '%';
  }
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const pct = (v) => (clamp01(v) * 100).toFixed(2) + '%';
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  // ------------------------------------------------------------------ //
  // Toasts & tooltip
  // ------------------------------------------------------------------ //
  function toast(message, kind = 'info', ms = 6000) {
    const host = $('#toasts');
    const el = h('div', { class: 'toast toast--' + kind },
      h('span', { class: 'toast__tag', text: kind === 'error' ? 'Error' : kind === 'ok' ? 'Done' : 'Note' }),
      h('span', { text: message }),
      h('button', { class: 'toast__close', type: 'button', 'aria-label': 'Dismiss', on: { click: () => el.remove() } }, icon('cross')));
    host.append(el);
    while (host.children.length > 4) host.firstChild.remove();
    setTimeout(() => el.remove(), ms);
  }
  function setStatus(el, text, kind) {
    if (!el) return;
    setText(el, text || '');
    const cls = 'statusline' + (kind ? ' statusline--' + kind : '');
    if (el.className !== cls) el.className = cls;
  }
  const tip = {
    key: '', width: 0, height: 0,
    show(evt, title, rows) {
      const el = $('#tooltip');
      const key = JSON.stringify([title, rows]);
      if (el.hidden || key !== this.key) {
        this.key = key;
        fill(el, title ? h('div', { class: 'tooltip__title', text: title }) : null,
          rows.map((r) => h('div', { class: 'tooltip__row' },
            r.cls ? h('span', { class: 'swatch ' + r.cls }) : null,
            h('span', { text: r.name }), h('strong', { text: r.value }))));
        el.hidden = false;
        this.width = el.offsetWidth; this.height = el.offsetHeight;
      }
      const pad = 12, w = this.width, ht = this.height;
      let x = evt.clientX + pad, y = evt.clientY + pad;
      if (x + w > innerWidth - 4) x = evt.clientX - w - pad;
      if (y + ht > innerHeight - 4) y = evt.clientY - ht - pad;
      el.style.left = Math.max(4, x) + 'px';
      el.style.top = Math.max(4, y) + 'px';
    },
    hide() { $('#tooltip').hidden = true; },
  };

  // ------------------------------------------------------------------ //
  // State & API
  // ------------------------------------------------------------------ //
  const S = {
    config: null,
    thresholds: { tau_high: 0.78, tau_low: 0.48, margin: 0.08, q_min: 0.4, l_min: 0.5 },
    engineReady: false, engineError: null, serverUp: true,
    subjects: [], tab: 'live', mode: 'identify', claimed: '',
    last: [null, null], lastAt: [0, 0],
  };
  const listeners = {};
  const on = (evt, fn) => { (listeners[evt] = listeners[evt] || []).push(fn); };
  const emit = (evt, arg) => { for (const fn of listeners[evt] || []) { try { fn(arg); } catch (e) { console.error(e); } } };

  const KEY_STORE = 'ffg.apiKey';
  function getKey() { try { return sessionStorage.getItem(KEY_STORE) || ''; } catch (e) { return memoryKey; } }
  let memoryKey = '';
  function setKey(k) { memoryKey = k; try { sessionStorage.setItem(KEY_STORE, k); } catch (e) { /* storage unavailable */ } }

  class ApiError extends Error {
    constructor(status, message, data) { super(message); this.status = status; this.data = data; }
  }
  function detailOf(data) {
    if (!data) return '';
    if (typeof data === 'string') return data.slice(0, 300);
    const d = data.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((x) => `${(x.loc || []).slice(1).join('.')}: ${x.msg}`).join('; ');
    return '';
  }
  async function apiRaw(path, opts = {}) {
    const headers = new Headers();
    const key = getKey();
    if (key) headers.set('X-API-Key', key);
    let body = opts.body;
    if (opts.json !== undefined) { headers.set('Content-Type', 'application/json'); body = JSON.stringify(opts.json); }
    let res;
    try {
      res = await fetch(path, { method: opts.method || 'GET', headers, body, signal: opts.signal, cache: 'no-store' });
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      throw new ApiError(0, 'The server could not be reached');
    }
    let data = null;
    if (opts.blob && res.ok) data = await res.blob();
    else {
      const text = await res.text();
      try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
    }
    if (res.status === 401) requestKey(key ? 'That key was not accepted.' : '');
    return { status: res.status, ok: res.ok, data };
  }
  async function api(path, opts) {
    const r = await apiRaw(path, opts);
    if (!r.ok) throw new ApiError(r.status, detailOf(r.data) || `Request failed (HTTP ${r.status})`, r.data);
    return r.data;
  }
  const wsUrl = (path) => (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + path;
  const backoff = (attempt) => Math.min(15000, 1000 * 2 ** attempt) + Math.round(Math.random() * 400);

  function requestKey(error) {
    const dlg = $('#key-dialog');
    $('#key-error').textContent = error || '';
    if (!dlg.open) { $('#key-input').value = ''; dlg.showModal(); }
  }

  // ------------------------------------------------------------------ //
  // Small UI components
  // ------------------------------------------------------------------ //
  function sliderField({ key, label, hint, min = 0, max = 1, step = 0.01, value = 0, onInput }) {
    const id = 'sl-' + key + '-' + Math.random().toString(36).slice(2, 7);
    const range = h('input', { type: 'range', id, min, max, step, value: String(value) });
    const number = h('input', { type: 'number', class: 'input--num', min, max, step, value: String(value), 'aria-label': label + ' (number)' });
    const sync = (src, dst) => () => {
      if (src.value === '' || isNaN(Number(src.value))) return;
      dst.value = src.value;
      if (onInput) onInput(Number(src.value));
    };
    range.addEventListener('input', sync(range, number));
    number.addEventListener('input', sync(number, range));
    const paint = () => {
      const lo = Number(min), hi = Number(max), v = Number(range.value);
      range.style.setProperty('--fill', `${hi > lo ? ((v - lo) / (hi - lo)) * 100 : 0}%`);
    };
    range.addEventListener('input', paint);
    number.addEventListener('input', paint);
    const el = h('div', { class: 'slider' },
      h('div', { class: 'slider__head' }, h('label', { for: id, class: 'slider__name', text: label }), number),
      hint ? h('p', { class: 'slider__hint', text: hint }) : null,
      range);
    paint();
    return { el, key, get: () => Number(number.value), set: (v) => { range.value = String(v); number.value = String(v); paint(); } };
  }
  const THRESHOLD_FIELDS = [
    { key: 'tau_high', label: 'Match threshold', hint: 'Scores at or above this confirm a match' },
    { key: 'tau_low', label: 'No-match threshold', hint: 'Scores below this are a clear no match' },
    { key: 'margin', label: 'Lookalike margin', hint: 'How far the best match must lead the next one' },
    { key: 'q_min', label: 'Minimum image quality', hint: 'Unclear images are sent for review' },
    { key: 'l_min', label: 'Minimum liveness', hint: 'Below this the face may be a photo or screen' },
  ];

  /* Horizontal 0..1 track with the three decision bands and score markers. */
  function createGauge(caption) {
    const bands = ['nomatch', 'uncertain', 'match'].map((k, i) =>
      h('div', { class: 'gauge__band gauge__band--' + k }, h('span', { text: ['No match', 'Needs review', 'Match'][i] })));
    const markers = ['s1', 's2'].map((key) => {
      const marker = h('div', { class: 'gauge__marker gauge__marker--' + key }, h('span', {}));
      const offset = h('div', { class: 'gauge__position' }, marker);
      return { marker, offset, position: h('div', { class: 'gauge__position', hidden: true }, offset) };
    });
    const tLow = h('span', { class: 'gauge__tick' }), tHigh = h('span', { class: 'gauge__tick' });
    const el = h('div', { class: 'gauge' },
      caption ? h('div', { class: 'label', text: caption }) : null,
      h('div', { class: 'gauge__plot' }, h('div', { class: 'gauge__track' }, bands), markers.map((m) => m.position)),
      h('div', { class: 'gauge__axis' }, h('span', { class: 'gauge__end', text: '0' }), tLow, tHigh, h('span', { class: 'gauge__end gauge__end--r', text: '1' })),
      h('ul', { class: 'gauge__legend', 'aria-hidden': 'true' },
        [['sw-nomatch', 'No match'], ['sw-uncertain', 'Needs review'], ['sw-match', 'Match']].map(([c, t]) => h('li', {}, h('span', { class: 'swatch ' + c }), t))));
    let lastLow, lastHigh;
    function mark(m, label, v, zoneChanged) {
      const ok = typeof v === 'number' && isFinite(v);
      if (m.position.hidden === ok) m.position.hidden = !ok;
      if (!ok) { m.effect?.cancel(); m.score = undefined; return; }
      const score = clamp01(v);
      if (score !== m.score) m.position.style.transform = `translateX(${pct(score)})`;
      if (zoneChanged) {
        m.effect?.cancel();
        if (m.score !== undefined && m.score !== score) {
          m.effect = animate(m.offset, [
            { transform: `translateX(${(m.score - score) * 100}%)` },
            { transform: 'translateX(0)' },
          ]);
        }
      }
      m.score = score;
      setText(m.marker.firstChild, `${label} ${v.toFixed(2)}`);
      m.marker.classList.toggle('gauge__marker--flip', score > 0.8);
    }
    return {
      el,
      set(th, s1, s2, labels = ['Best', 'Next'], zoneChanged = false) {
        const lo = clamp01(th.tau_low), hi = clamp01(th.tau_high);
        if (lo !== lastLow || hi !== lastHigh) {
          lastLow = lo; lastHigh = hi;
          bands[0].style.width = pct(lo);
          bands[1].style.left = pct(lo); bands[1].style.width = pct(hi - lo);
          bands[2].style.left = pct(hi); bands[2].style.width = pct(1 - hi);
          tLow.style.left = pct(lo); tLow.textContent = lo.toFixed(2);
          tHigh.style.left = pct(hi); tHigh.textContent = hi.toFixed(2);
        }
        mark(markers[0], labels[0], s1, zoneChanged);
        mark(markers[1], labels[1], s2, zoneChanged);
      },
    };
  }

  function createMeter(label, markLabel) {
    const value = h('span', { class: 'meter__value', text: '–' });
    const fillEl = h('div', { class: 'meter__fill' });
    const markEl = h('div', { class: 'meter__mark', hidden: true });
    const el = h('div', { class: 'meter' },
      h('div', { class: 'meter__head' }, h('span', { text: label }), value),
      h('div', { class: 'meter__track' }, fillEl, markEl));
    return {
      el,
      set(v, mark, note) {
        const ok = typeof v === 'number' && isFinite(v);
        const scale = `scaleX(${ok ? clamp01(v) : 0})`;
        if (fillEl.style.transform !== scale) fillEl.style.transform = scale;
        const hasMark = typeof mark === 'number';
        if (markEl.hidden === hasMark) markEl.hidden = !hasMark;
        if (hasMark && markEl.title !== `${markLabel || 'minimum'} ${mark.toFixed(2)}`) {
          markEl.style.left = pct(mark); markEl.title = `${markLabel || 'minimum'} ${mark.toFixed(2)}`;
        }
        const below = ok && hasMark && v < mark;
        const level = !ok ? 'none' : !hasMark ? 'plain' : below ? 'low' : 'ok';
        if (el.dataset.level !== level) el.dataset.level = level;
        let text = ok ? v.toFixed(2) : '–';
        if (hasMark && ok) text = below ? `${text} · too low` : `${text} · good`;
        setText(value, note || text);
      },
    };
  }

  function verdictBlock(announceLabel) {
    const iconBox = h('span', { class: 'verdict__icon' });
    const kicker = h('div', { class: 'verdict__kicker' });
    const title = h('div', { class: 'verdict__title' });
    const reason = h('div', { class: 'verdict__reason' });
    const announcement = announceLabel ? h('p', { class: 'visually-hidden', role: 'status', 'aria-atomic': 'true' }) : null;
    const text = h('div', { class: 'verdict__text' }, kicker, title, reason);
    const el = h('div', { class: 'verdict', data: { zone: 'NONE' } }, iconBox, text, announcement);
    let shown = '', pending = '', announceTimer, effect, lastMotion = -Infinity;
    return {
      el,
      set(zone, kickerText, titleText, reasonText) {
        const changed = zone !== shown;
        const now = performance.now();
        const animateChange = changed && shown && now - lastMotion >= standardMotion;
        if (changed) {
          fill(iconBox, icon(ZONES[zone] ? ZONES[zone].icon : 'minus')); el.dataset.zone = zone;
          if (animateChange) {
            lastMotion = now;
            effect?.cancel();
            effect = animate(text, [{ opacity: .55 }, { opacity: 1 }]);
          }
          shown = zone;
        }
        setText(kicker, kickerText || '');
        setText(title, titleText || '');
        setText(reason, reasonText || '');
        if (announcement) {
          const label = ZONES[zone]?.label || titleText;
          const name = zone === 'MATCH_CONFIRMED' && titleText !== label ? `. ${titleText}` : '';
          const next = `${announceLabel}: ${label}${name}`;
          if (next !== pending) {
            pending = next;
            clearTimeout(announceTimer);
            announceTimer = setTimeout(() => {
              if (!el.closest('[hidden]')) setText(announcement, next);
            }, 400);
          }
        }
        return Boolean(animateChange);
      },
      decision(decision, prefix) {
        const v = verdictOf(decision);
        const extra = (decision.reasons || []).slice(1).map(reasonLabel);
        return this.set(v.zone, [prefix, v.kicker].filter(Boolean).join(' · '), v.title, [v.reason, ...extra].join('; '));
      },
    };
  }

  // ------------------------------------------------------------------ //
  // Video stage with per-face overlay
  // ------------------------------------------------------------------ //
  function drawIcon(ctx, name, x, y, size) {
    ctx.save();
    ctx.translate(x, y); ctx.scale(size / 24, size / 24);
    ctx.lineWidth = 2.6; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
    for (const d of ICONS[name] || []) ctx.stroke(new Path2D(d));
    ctx.restore();
  }
  function createStage({ label, mirror = false, engineTag = '', image = false }) {
    const video = image ? h('img', { alt: label + ' preview' }) : h('video', { class: mirror ? 'mirror' : '', autoplay: true, playsinline: true });
    video.muted = true;
    const canvas = h('canvas', { class: 'stage__overlay', 'aria-hidden': 'true' });
    const placeholder = h('div', { class: 'stage__placeholder' }, icon('camera'), h('span', { text: 'Camera is off' }));
    const note = h('div', { class: 'stage__note', hidden: true });
    const el = h('div', { class: 'stage' }, video, canvas, placeholder,
      h('div', { class: 'stage__label', text: label }), note);
    let last = null, engine = 'fair', width = 0, height = 0, aspect = '';
    const ctx = canvas.getContext('2d');
    function draw() {
      const cw = window.ResizeObserver ? width : el.clientWidth;
      const ch = window.ResizeObserver ? height : el.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (!cw || !ch) return;
      if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
        canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      if (!last || !last.frame) { note.hidden = true; return; }
      const faces = last.faces || [];
      note.hidden = faces.length > 0;
      if (!faces.length) { setText(note, 'No face detected'); return; }
      const fw = last.frame.width, fh = last.frame.height;
      const scale = Math.min(cw / fw, ch / fh), ox = (cw - fw * scale) / 2, oy = (ch - fh * scale) / 2;
      const ordered = faces.slice().sort((a, b) => Number(a.primary) - Number(b.primary));
      for (const face of ordered) {
        const decision = engine === 'baseline' ? face.baseline : face.decision;
        if (!decision) continue;
        const v = verdictOf(decision);
        // A confirmed match with someone on the banned list is drawn in red, by name.
        const banned = engine !== 'baseline' && isBanned(v.name);
        const z = banned ? BANNED_STYLE : ZONES[v.zone];
        const [x1, y1, x2, y2] = face.bbox;
        const w = (x2 - x1) * scale, ht = (y2 - y1) * scale, y = oy + y1 * scale;
        let x = ox + x1 * scale;
        if (mirror) x = cw - x - w; // the video is flipped with CSS; flip geometry only, text stays readable
        ctx.save();
        ctx.globalAlpha = face.primary ? 1 : 0.85;
        ctx.lineWidth = face.primary ? 3 : 1.75;
        ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.setLineDash([]);
        ctx.strokeRect(x - 1, y - 1, w + 2, ht + 2);
        ctx.strokeStyle = z.box; ctx.setLineDash(z.dash);
        ctx.strokeRect(x, y, w, ht);
        ctx.setLineDash([]);
        const fs = face.primary ? 13 : 11.5;
        ctx.font = `600 ${fs}px system-ui, -apple-system, "Segoe UI", sans-serif`;
        let text = banned ? `${v.name} - Banned` : (v.name || v.visitor || z.label);
        if (engineTag) text = `${engine === 'baseline' ? 'Conventional' : 'Fairview'}: ${text}`;
        const maxW = Math.max(90, cw - 16);
        while (ctx.measureText(text).width > maxW - 34 && text.length > 6) text = text.slice(0, -2).trimEnd() + '…';
        const bw = ctx.measureText(text).width + fs + 20, bh = fs + 11;
        const bx = Math.max(4, Math.min(cw - bw - 4, x));
        const by = y - bh - 4 >= 2 ? y - bh - 4 : Math.min(ch - bh - 2, y + 4);
        ctx.fillStyle = z.badge;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 5); else ctx.rect(bx, by, bw, bh);
        ctx.fill();
        drawIcon(ctx, z.icon, bx + 6, by + (bh - fs) / 2, fs);
        ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle';
        ctx.fillText(text, bx + fs + 12, by + bh / 2 + 0.5);
        ctx.restore();
      }
    }
    if (window.ResizeObserver) new ResizeObserver(([entry]) => {
      ({ width, height } = entry.contentRect);
      draw();
    }).observe(el);
    video.addEventListener('loadedmetadata', () => {
      if (video.videoWidth) el.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      draw();
    });
    return {
      el, video,
      setStream(stream, offText) {
        if (image) { if (stream) video.src = stream; else video.removeAttribute('src'); }
        else if (video.srcObject !== stream) video.srcObject = stream || null;
        placeholder.hidden = !!stream;
        if (offText) placeholder.lastChild.textContent = offText;
        if (stream && !image) video.play().catch(() => {});
        if (!stream) { last = null; draw(); }
      },
      show(result, eng) {
        last = result; engine = eng || 'fair';
        const frame = result?.frame;
        if (image && frame?.width && frame?.height) {
          const next = `${frame.width} / ${frame.height}`;
          if (aspect !== next) { aspect = next; el.style.aspectRatio = aspect; }
        }
        draw();
      },
    };
  }

  // ------------------------------------------------------------------ //
  // Primary-face analysis panel
  // ------------------------------------------------------------------ //
  function createFacePanel(title) {
    const verdict = verdictBlock(title);
    const facesNote = h('p', { class: 'muted small' });
    const gauge = createGauge('Match confidence');
    const margin = h('p', { class: 'readout' });
    const quality = createMeter('Image quality', 'minimum');
    const qNorm = createMeter('Sharpness'), qDet = createMeter('Detection'), qPose = createMeter('Head angle');
    const featNorm = h('span', { class: 'muted small' });
    const liveness = createMeter('Liveness', 'minimum');
    const wAda = createMeter('Model blend');
    const timing = h('dl', { class: 'kv kv--tight' });
    const candTitle = h('div', { class: 'label' });
    const candList = h('ol', { class: 'cand-list' });
    const details = h('details', { class: 'disclosure panel__details' }, h('summary', { text: 'Technical details' }),
      h('div', {}, margin,
        h('div', { class: 'panel__group' }, h('div', { class: 'label', text: 'Image quality breakdown' }), h('div', { class: 'meter-sub' }, qNorm.el, qDet.el, qPose.el), featNorm),
        wAda.el,
        h('div', { class: 'panel__group' }, candTitle, candList),
        h('div', { class: 'panel__group' }, h('div', { class: 'label', text: 'Processing time' }), timing)));
    const body = h('div', { class: 'panel__body' },
      facesNote, gauge.el,
      h('div', { class: 'panel__group' }, quality.el, liveness.el),
      details);
    // Shown when the matched or closest person is on the site's banned list.
    const bannedTitle = h('strong', {}), bannedNote = h('span', {});
    const bannedAlert = h('div', { class: 'panel__alert', role: 'status', hidden: true }, bannedTitle, bannedNote);
    const el = h('section', { class: 'card panel', 'aria-label': title },
      h('h2', { class: 'card__title', text: title }), verdict.el, bannedAlert, body);
    let latest;
    function idle(text, sub) { latest = null; verdict.set('NONE', '', text, sub || ''); body.hidden = true; bannedAlert.hidden = true; verdict.el.dataset.banned = 'false'; }
    function showBanned(d) {
      const watched = d.zone === 'MATCH_CONFIRMED' || d.zone === 'UNCERTAIN_REVIEW';
      const hit = watched ? S.subjects.find((x) => x.banned && (x.subject === d.subject || x.subject === (d.top1 && d.top1.subject))) : null;
      bannedAlert.hidden = !hit;
      // A confirmed banned match turns the verdict red as well as the camera box.
      verdict.el.dataset.banned = hit && d.zone === 'MATCH_CONFIRMED' ? 'true' : 'false';
      if (!hit) return;
      setText(bannedTitle, `${d.zone === 'MATCH_CONFIRMED' ? '' : 'Possibly '}${hit.subject} · banned from premises`);
      setText(bannedNote, `${hit.banned_note ? hit.banned_note + '. ' : ''}From this site’s own list. Sent to the review queue for a guard to check; nothing happens automatically.`);
    }
    function showTiming(msg, fps) {
      const t = (msg && msg.timing_ms) || {};
      fill(timing, [['detect', 'Detect'], ['liveness', 'Liveness'], ['embed', 'Encode'], ['decide', 'Decide'], ['total', 'Total']].map(([k, label]) =>
        [h('dt', { text: label }), h('dd', { text: typeof t[k] === 'number' ? t[k].toFixed(0) + ' ms' : '–' })]),
        h('dt', { text: 'Frame rate' }), h('dd', { text: fps ? fps.toFixed(1) + ' fps' : '–' }),
        msg && msg.dropped ? [h('dt', { text: 'Skipped frames' }), h('dd', { text: String(msg.dropped) })] : null);
    }
    details.addEventListener('toggle', () => {
      if (details.open && latest) renderDetails(...latest);
    });
    function renderDetails(msg, fps, face, d, th, q) {
      showTiming(msg, fps);
      if (d.mode === 'verify') { setText(margin, ''); }
      else if (typeof d.margin === 'number') {
        const narrow = d.margin < th.margin;
        setText(margin, `Lead over the next closest person: ${d.margin.toFixed(2)}` +
          (narrow ? ` (needs ${th.margin.toFixed(2)}, too close to tell apart)` : ' (clear lead)'));
        margin.dataset.level = narrow ? 'low' : 'ok';
      } else { setText(margin, 'Only one person to compare against.'); margin.dataset.level = 'none'; }
      qNorm.set(q.norm); qDet.set(q.detection); qPose.set(q.pose);
      setText(featNorm, `Feature strength ${num(q.feature_norm, 1)} · detector confidence ${num(face.det_score, 2)}`);
      if (typeof face.w_ada === 'number') wAda.set(face.w_ada, undefined, `AdaFace ${Math.round(face.w_ada * 100)}% · ArcFace ${Math.round((1 - face.w_ada) * 100)}%`);
      else wAda.set(null);
      const list = face.candidates || [];
      const confirmed = d.zone === 'MATCH_CONFIRMED';
      setText(candTitle, confirmed ? 'Closest enrolled people' : 'Closest enrolled people (unconfirmed)');
      fill(candList, list.length ? list.map((c) => h('li', {},
        h('span', { class: 'cand__name', text: c.subject }),
        h('span', { class: 'muted', text: num(c.score, 2) })))
        : h('li', { class: 'muted', text: 'No one is enrolled yet.' }));
    }
    idle('Camera is off', 'Start the camera to begin.');
    return {
      el, idle,
      update(msg, fps) {
        const faces = msg.faces || [];
        if (!faces.length) { idle('No face detected', 'Make sure a face is clearly in view.'); return; }
        const face = faces.find((f) => f.primary) || faces[0];
        const d = face.decision, th = d.thresholds || S.thresholds;
        body.hidden = false;
        const changed = verdict.decision(d, d.mode === 'verify' ? `Checking: ${msg.claimed_subject || ''}` : '');
        showBanned(d);
        const note = face.error ? `This face could not be analyzed: ${humanize(face.error).toLowerCase()}.`
          : faces.length > 1 ? `${faces.length} faces in view. The result shows the largest face. Labels appear on the video.` : '';
        setText(facesNote, note);
        facesNote.hidden = !note;
        gauge.set(th, d.top1?.score, d.top2?.score, undefined, changed);
        const q = face.quality || {};
        quality.set(q.score, th.q_min);
        const lv = face.liveness || {};
        if (!lv.enabled) liveness.set(null, undefined, 'Off');
        else if (typeof lv.score !== 'number') liveness.set(null, undefined, 'Unavailable');
        else liveness.set(lv.score, th.l_min);
        latest = [msg, fps, face, d, th, q];
        if (details.open) renderDetails(...latest);
      },
    };
  }

  // ------------------------------------------------------------------ //
  // Camera
  // ------------------------------------------------------------------ //
  const cam = { stream: null, starting: false };
  const capVideo = [0, 1].map(() => { const v = h('video', { autoplay: true, playsinline: true }); v.muted = true; return v; });
  const grabCanvas = [0, 1, 2].map(() => document.createElement('canvas'));
  const localStages = [];

  function cameraErrorText(err) {
    const name = err && err.name;
    if (name === 'NotAllowedError' || name === 'SecurityError') return 'Camera permission was denied. Allow camera access for this site in the browser settings, then try again.';
    if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.';
    if (name === 'NotReadableError' || name === 'AbortError') return 'The camera is in use by another application or could not be started.';
    return 'The camera could not be started' + (err && err.message ? `: ${err.message}` : '.');
  }
  async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera access is unavailable in this context. Browsers only expose the camera on HTTPS or on localhost — open this page through an https:// address.');
    }
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } });
    } catch (err) { throw new Error(cameraErrorText(err)); }
  }
  async function startCamera() {
    if (cam.stream || cam.starting) return !!cam.stream;
    cam.starting = true; emit('camera');
    try {
      const stream = await openCamera();
      cam.stream = stream;
      const track = stream.getVideoTracks()[0];
      if (track) track.addEventListener('ended', () => { if (cam.stream === stream) { stopCamera(); toast('The camera stopped delivering video.', 'error'); } });
      setStatus($('#live-status'), '');
      return true;
    } catch (err) {
      setStatus($('#live-status'), err.message, 'error');
      toast(err.message, 'error', 10000);
      return false;
    } finally { cam.starting = false; emit('camera'); }
  }
  function stopCamera() {
    if (!cam.stream) return;
    cam.stream.getTracks().forEach((t) => t.stop());
    cam.stream = null;
    emit('camera');
  }
  function grab(video, canvas, maxW, quality) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh || video.readyState < 2) return Promise.resolve(null);
    const k = Math.min(1, maxW / vw);
    canvas.width = Math.round(vw * k); canvas.height = Math.round(vh * k);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
  }

  // ------------------------------------------------------------------ //
  // /ws/camera: frames out, decisions out, reconnect with backoff
  // ------------------------------------------------------------------ //
  const link = { ws: null, attempt: 0, timer: null, busy: [false, false], watchdog: [0, 0], retry: [0, 0], sentAt: [0, 0], stamps: [[], []], blocked: '' };
  const MIN_FRAME_INTERVAL = 66;
  const streamActive = (id) => !!capVideo[id].srcObject;
  const wantLink = () => streamActive(0) || streamActive(1);

  function linkStatus(text, kind) {
    for (const id of ['#live-status', '#ab-status']) setStatus($(id), text, kind);
  }
  function effectiveMode() {
    if (S.tab === 'stress') return { mode: 'identify', claimed: null };
    if (S.mode === 'verify' && S.claimed) return { mode: 'verify', claimed: S.claimed };
    return { mode: 'identify', claimed: null };
  }
  function sendLinkConfig() {
    if (!link.ws || link.ws.readyState !== 1) return;
    const m = effectiveMode();
    link.ws.send(JSON.stringify({ type: 'config', mode: m.mode, claimed_subject: m.claimed, compare_baseline: S.tab === 'ab' }));
  }
  function ensureLink() {
    if (!wantLink()) {
      clearTimeout(link.timer);
      if (link.ws) { const ws = link.ws; link.ws = null; ws.close(1000); }
      return;
    }
    if (link.ws || link.blocked) return;
    clearTimeout(link.timer);
    const key = getKey();
    let ws;
    try { ws = new WebSocket(wsUrl('/ws/camera')); }
    catch (e) { linkStatus('Recognition link could not be opened.', 'error'); return; }
    link.ws = ws;
    ws.addEventListener('open', () => {
      // The key goes in the first message, never in the URL (URLs end up in logs).
      if (key) ws.send(JSON.stringify({ type: 'auth', api_key: key }));
      link.attempt = 0; linkStatus(''); sendLinkConfig(); pump(0); pump(1);
    });
    ws.addEventListener('message', onLinkMessage);
    ws.addEventListener('close', (ev) => {
      if (link.ws !== ws) return;
      link.ws = null; link.busy = [false, false];
      if (ev.code === 4401) { link.blocked = 'key'; linkStatus('Recognition link refused: API key missing or invalid.', 'error'); requestKey(getKey() ? 'That key was not accepted.' : ''); return; }
      if (ev.code === 4403) { link.blocked = 'origin'; linkStatus('Recognition link refused: the page origin does not match the server address.', 'error'); return; }
      if (!wantLink()) return;
      const delay = backoff(link.attempt++);
      linkStatus(`Recognition link lost. Reconnecting in ${Math.round(delay / 1000)} s…`, 'error');
      link.timer = setTimeout(ensureLink, delay);
    });
    ws.addEventListener('error', () => { /* a close event follows and drives the reconnect */ });
  }
  function later(id, ms) { clearTimeout(link.retry[id]); link.retry[id] = setTimeout(() => pump(id), ms); }
  function frameDone(id, delay) {
    clearTimeout(link.watchdog[id]);
    link.busy[id] = false;
    if (delay) later(id, delay); else pump(id);
  }
  /* One frame in flight per stream: the next is sent only after the reply for
   * the previous one arrived (or a 1.5 s watchdog fired). */
  function pump(id) {
    if (link.busy[id] || !link.ws || link.ws.readyState !== 1 || !streamActive(id) || !S.engineReady) return;
    const since = performance.now() - link.sentAt[id];
    if (since < MIN_FRAME_INTERVAL) { later(id, MIN_FRAME_INTERVAL - since); return; }
    link.busy[id] = true;
    grab(capVideo[id], grabCanvas[id], 640, 0.8).then((blob) => {
      const ws = link.ws;
      if (!blob || !ws || ws.readyState !== 1) { link.busy[id] = false; later(id, 250); return; }
      ws.send(new Blob([new Uint8Array([id]), blob]));
      link.sentAt[id] = performance.now();
      clearTimeout(link.watchdog[id]);
      link.watchdog[id] = setTimeout(() => { link.busy[id] = false; pump(id); }, 1500);
    }).catch(() => { link.busy[id] = false; later(id, 500); });
  }
  function onLinkMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.type === 'ready') {
      if (msg.thresholds) { S.thresholds = msg.thresholds; emit('thresholds'); }
    } else if (msg.type === 'result') {
      const id = msg.stream === 'remote' ? 1 : 0;
      const now = performance.now(), stamps = link.stamps[id];
      stamps.push(now);
      while (stamps.length > 2 && now - stamps[0] > 3000) stamps.shift();
      const fps = stamps.length > 1 ? ((stamps.length - 1) * 1000) / (now - stamps[0]) : 0;
      S.last[id] = msg; S.lastAt[id] = now;
      if (link.errorShown) { link.errorShown = false; linkStatus(''); }
      emit('result', { id, msg, fps });
      frameDone(id);
    } else if (msg.type === 'error') {
      if (msg.code === 'unauthorized') return; // the close handler prompts for the key
      const text = msg.code === 'unknown-subject' ? 'The selected person is not enrolled. Choose another person or switch to Identify.'
        : msg.code === 'incomplete-template' ? 'This person’s enrollment is incomplete. Capture more samples before verifying them.'
        : `Frame could not be processed: ${msg.message || msg.code}`;
      link.errorShown = true;
      linkStatus(text, 'error');
      if (msg.stream) frameDone(msg.stream === 'remote' ? 1 : 0, 1000);
    }
  }
  setInterval(() => { // safety net and stale-result sweep
    pump(0); pump(1);
    for (const id of [0, 1]) {
      if (S.last[id] && performance.now() - S.lastAt[id] > 2500) { S.last[id] = null; link.stamps[id] = []; emit('stale', id); }
    }
  }, 1000);

  // ------------------------------------------------------------------ //
  // Enrollment capture (shared by the live view and the stress test)
  // ------------------------------------------------------------------ //
  const GUIDANCE = ['Look straight at the camera', 'Turn your head slightly to the left', 'Turn your head slightly to the right',
    'Raise your chin slightly', 'Look straight at the camera again'];
  function createEnroller({ defaultName = '', allowSource = false, onDone, checkIn = false } = {}) {
    const uid = Math.random().toString(36).slice(2, 7);
    const name = h('input', { type: 'text', id: 'en-name-' + uid, maxlength: 80, autocomplete: 'off', spellcheck: 'false', value: defaultName });
    const count = h('select', { id: 'en-count-' + uid });
    const replace = h('input', { type: 'checkbox' });
    const source = h('select', { id: 'en-src-' + uid }, h('option', { value: '0', text: 'My camera' }), h('option', { value: '1', text: 'Remote participant' }));
    const button = h('button', { class: 'btn btn--primary', type: 'submit', text: checkIn ? 'Capture check-in' : 'Capture and add' });
    const progress = h('progress', { max: 5, value: 0, hidden: true, 'aria-label': 'Capture progress' });
    const status = h('p', { class: 'statusline', role: 'status', 'aria-live': 'polite' });
    const result = h('div', { class: 'enroll-result' });
    const sourceField = h('label', { class: 'field', for: source.id, hidden: !allowSource }, h('span', { text: 'Camera' }), source);
    name.placeholder = 'e.g. Jordan Lee';
    const el = h('form', { class: 'enroller', novalidate: true },
      h('label', { class: 'field', for: name.id }, h('span', { text: 'Full name' }), name),
      h('div', { class: 'field-row' },
        h('label', { class: 'field', for: count.id }, h('span', { text: 'Photos to take' }), count),
        sourceField),
      h('label', { class: 'check' }, replace, h('span', { text: 'Replace this person’s existing photos' })),
      h('p', { class: 'hint', text: checkIn ? 'Returning for a check-in? Use the same name.' : 'Look at the camera and turn your head slightly between photos.' }),
      h('div', { class: 'row' }, button, progress), status, result);
    function fillCounts() {
      const e = (S.config && S.config.enrollment) || { min_samples: 3, max_samples: 5 };
      const prev = Number(count.value) || e.max_samples;
      fill(count, Array.from({ length: e.max_samples - e.min_samples + 1 }, (_, i) => h('option', { text: String(e.min_samples + i) })));
      count.value = String(Math.min(e.max_samples, Math.max(e.min_samples, prev)));
    }
    fillCounts(); on('config', fillCounts);
    let running = false;
    el.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      if (running) return;
      const subject = name.value.trim();
      const src = allowSource ? Number(source.value) : 0;
      if (!subject) { setStatus(status, 'Enter a name first.', 'error'); name.focus(); return; }
      if (!streamActive(src)) { setStatus(status, src ? 'No remote participant video is available.' : 'Start the camera first.', 'error'); return; }
      running = true; button.disabled = true; clear(result);
      const n = Number(count.value) || 3;
      progress.max = n; progress.value = 0; progress.hidden = false;
      try {
        const form = new FormData();
        form.append('subject', subject);
        form.append('replace', replace.checked ? 'true' : 'false');
        let got = 0;
        for (let i = 0; i < n; i++) {
          setStatus(status, `Photo ${i + 1} of ${n}: ${GUIDANCE[i % GUIDANCE.length]}…`);
          await wait(i === 0 ? 700 : 400);
          const blob = await grab(capVideo[src], grabCanvas[2], 960, 0.9);
          if (blob) { form.append('files', blob, `frame-${i + 1}.jpg`); got++; }
          progress.value = i + 1;
        }
        if (!got) throw new Error('No photos could be taken from the video.');
        setStatus(status, 'Checking photos…');
        const r = await apiRaw('/api/enroll', { method: 'POST', body: form });
        if (r.data && typeof r.data === 'object' && Array.isArray(r.data.frames)) {
          renderEnrollResult(result, r.data);
          setStatus(status, r.data.enrolled ? '' : 'Not added: none of the photos could be used.', r.data.enrolled ? 'ok' : 'error');
          if (r.data.enrolled) { await loadSubjects(); if (onDone) onDone(r.data.subject); }
        } else throw new Error(detailOf(r.data) || `Enrollment failed (HTTP ${r.status})`);
      } catch (err) {
        setStatus(status, err.message, 'error');
      } finally { running = false; button.disabled = false; progress.hidden = true; }
    });
    return { el, setName: (v) => { name.value = v; }, focus: () => { name.focus(); el.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } };
  }
  function renderEnrollResult(host, data) {
    const e = (S.config && S.config.enrollment) || {};
    const rec = data.recommended_samples || [e.min_samples || 3, e.max_samples || 5];
    const frames = data.frames || [];
    const rejected = frames.filter((f) => !f.accepted);
    fill(host,
      data.enrolled ? h('p', { class: 'notice ' + (data.template_complete ? 'notice--ok' : 'notice--warn') },
        h('strong', { text: data.subject }),
        data.template_complete && data.active !== false
          ? ` is checked in. ${data.accepted} of ${frames.length} photos were used.`
          : ` was saved, but needs at least ${rec[0]} good photos before they can be recognised (has ${data.samples}). Capture again to add more.`)
        : h('p', { class: 'notice notice--error', text: 'None of the photos could be used, so nothing was saved. Try better lighting and face the camera.' }),
      data.possible_duplicate_of ? h('p', { class: 'notice notice--warn' }, 'This looks like ',
        h('strong', { text: data.possible_duplicate_of }), ', who is already enrolled. Make sure the same person is not added twice.') : null,
      rejected.length ? h('details', { class: 'disclosure' }, h('summary', { text: `${rejected.length} photo${rejected.length === 1 ? ' was' : 's were'} not used` }),
        h('ul', { class: 'frame-list' }, rejected.map((f) => h('li', { data: { ok: 'false' } },
          icon('cross'),
          h('span', {}, h('strong', { text: `Photo ${f.index + 1}` }), ` — ${reasonLabel(f.reason)}`))))) : null);
  }

  // ------------------------------------------------------------------ //
  // Subjects
  // ------------------------------------------------------------------ //
  const subjectUi = { name: null, action: null, error: '' };
  async function loadSubjects() {
    try {
      const data = await api('/api/subjects');
      S.subjects = data.subjects || [];
      emit('subjects');
    } catch (err) { toast(`Subjects could not be loaded: ${err.message}`, 'error'); }
  }
  function fillSubjectSelect(select, placeholder, activeOnly) {
    const prev = select.value;
    const inactive = (x) => x.active === false || x.template_complete === false;
    fill(select, h('option', { value: '', text: placeholder }), S.subjects.map((x) => h('option', { value: x.subject,
      text: x.subject + (inactive(x) ? ' (incomplete template)' : ''), disabled: !!activeOnly && inactive(x) })));
    select.value = S.subjects.some((x) => x.subject === prev) ? prev : '';
  }
  /* Five quiet dots, one per check-in day, then "Onboarded". */
  function onboardingProgress(sub) {
    const target = (S.config && S.config.enrollment && S.config.enrollment.onboarding_days) || 5;
    const days = Math.min(target, Number(sub.check_in_days) || 0);
    const done = days >= target;
    return h('span', { class: 'onboard', data: { done: String(done) }, title: `${days} of ${target} check-in days` },
      h('span', { class: 'onboard__dots', 'aria-hidden': 'true' }, Array.from({ length: target }, (_, i) => h('span', { class: 'onboard__dot' + (i < days ? ' is-on' : '') }))),
      h('span', { class: 'onboard__text', text: done ? 'Onboarded' : `Check-in day ${days} of ${target}` }));
  }
  function renderSubjects(restore) {
    const list = $('#subject-list');
    $('#subjects-empty').hidden = S.subjects.length > 0;
    const min = (S.config && S.config.enrollment && S.config.enrollment.min_samples) || 3;
    let focusEl = null;
    fill(list, S.subjects.map((sub) => {
      const active = subjectUi.name === sub.subject ? subjectUi.action : null;
      const reset = () => {
        subjectUi.name = null; subjectUi.action = null; subjectUi.error = '';
        renderSubjects({ name: sub.subject, action: active });
      };
      const li = h('li', { class: 'subject' });
      const initials = sub.subject.trim().split(/\s+/).slice(0, 2).map((w) => w[0] || '').join('').toUpperCase();
      li.append(h('span', { class: 'subject__avatar', 'aria-hidden': 'true', text: initials || '?' }));
      const info = h('div', { class: 'subject__info' },
        h('span', { class: 'subject__name', text: sub.subject }),
        h('span', { class: 'muted small', text: `${sub.samples} photo${sub.samples === 1 ? '' : 's'}` +
          (typeof sub.mean_quality === 'number' ? ` · quality ${Math.round(sub.mean_quality * 100)}%` : '') }),
        sub.active === false || sub.template_complete === false
          ? h('span', { class: 'chip', data: { zone: 'UNCERTAIN_REVIEW' } }, icon('alert'), h('span', { text: `Needs ${Math.max(1, min - sub.samples)} more photo${min - sub.samples === 1 ? '' : 's'}` }))
          : sub.banned ? null : onboardingProgress(sub),
        sub.banned ? h('span', { class: 'chip', data: { zone: 'SPOOF_DETECTED' } }, icon('alert'), h('span', { text: 'Banned from premises' })) : null,
        sub.banned && (sub.banned_note || sub.banned_photo)
          ? h('span', { class: 'muted small', text: [sub.banned_note, sub.banned_photo ? 'Record photo on file' : null].filter(Boolean).join(' · ') }) : null);
      const subjectUrl = '/api/subjects/' + encodeURIComponent(sub.subject);
      if (active === 'ban') {
        const note = h('input', { type: 'text', maxlength: 300, value: sub.banned_note || '', placeholder: 'Reason or reference', 'aria-label': `Reason for banning ${sub.subject}` });
        const photo = h('input', { type: 'file', accept: 'image/*', 'aria-label': `Record photo of ${sub.subject} (optional)` });
        focusEl = note;
        li.append(info, h('form', { class: 'subject__ban', on: { submit: async (ev) => {
          ev.preventDefault();
          const body = new FormData();
          body.append('note', note.value.trim());
          if (photo.files && photo.files[0]) body.append('photo', photo.files[0]);
          try {
            await api(subjectUrl + '/ban', { method: 'POST', body });
            reviewImages.delete(recordKey(sub.subject)); // a new record photo may have replaced the old one
            toast(`${sub.subject} is on the banned list.`, 'ok'); subjectUi.name = null; subjectUi.error = '';
            await loadSubjects();
          } catch (err) { subjectUi.error = err.message; renderSubjects(); }
        } } },
        note,
        h('label', { class: 'subject__file' }, h('span', { text: 'Record photo (optional)' }), photo),
        h('p', { class: 'muted small', text: 'This is your site’s own list. If this person is seen, the case goes to the review queue for a guard to check. Nothing happens automatically.' }),
        h('div', { class: 'row' },
          h('button', { class: 'btn btn--small btn--danger', type: 'submit', text: sub.banned ? 'Save' : 'Ban from premises' }),
          h('button', { class: 'btn btn--small', type: 'button', text: 'Cancel', on: { click: reset } }))));
      } else if (active === 'unban') {
        const yes = h('button', { class: 'btn btn--small btn--primary', type: 'button', text: 'Lift ban', on: { click: async () => {
          try {
            await api(subjectUrl + '/ban', { method: 'DELETE' });
            toast(`The ban on ${sub.subject} was lifted.`, 'ok'); subjectUi.name = null; subjectUi.error = '';
            await loadSubjects();
          } catch (err) { subjectUi.error = err.message; renderSubjects(); }
        } } });
        focusEl = yes;
        li.append(info, h('div', { class: 'subject__confirm subject__confirm--plain', role: 'alert' },
          h('span', { text: sub.banned_photo ? 'Lift the ban and delete the record photo?' : 'Lift the ban?' }), yes,
          h('button', { class: 'btn btn--small', type: 'button', text: 'Cancel', on: { click: reset } })));
      } else if (active === 'rename') {
        const input = h('input', { type: 'text', value: sub.subject, maxlength: 80, 'aria-label': `New name for ${sub.subject}` });
        focusEl = input;
        li.append(h('form', { class: 'subject__edit', on: { submit: async (ev) => {
          ev.preventDefault();
          const next = input.value.trim();
          if (!next || next === sub.subject) { reset(); return; }
          try {
            await api('/api/subjects/' + encodeURIComponent(sub.subject), { method: 'PUT', json: { subject: next } });
            toast(`Renamed to ${next}.`, 'ok'); subjectUi.name = null; subjectUi.error = '';
            await loadSubjects();
            if (S.tab === 'people') focusSubjectAction(next, 'rename');
          } catch (err) { subjectUi.error = err.message; renderSubjects(); }
        } } }, input, h('button', { class: 'btn btn--small btn--primary', type: 'submit', text: 'Save' }),
        h('button', { class: 'btn btn--small', type: 'button', text: 'Cancel', on: { click: reset } })));
      } else if (active === 'delete') {
        const yes = h('button', { class: 'btn btn--small btn--danger', type: 'button', text: 'Delete', on: { click: async () => {
          try {
            await api('/api/subjects/' + encodeURIComponent(sub.subject), { method: 'DELETE' });
            toast(`${sub.subject} was removed.`, 'ok'); subjectUi.name = null; subjectUi.error = '';
            await loadSubjects();
            if (S.tab === 'people') $('#subjects-refresh').focus();
          } catch (err) { subjectUi.error = err.message; renderSubjects(); }
        } } });
        focusEl = yes;
        li.append(info, h('div', { class: 'subject__confirm', role: 'alert' },
          h('span', { text: 'Remove this person and their stored face data?' }), yes,
          h('button', { class: 'btn btn--small', type: 'button', text: 'Cancel', on: { click: reset } })));
      } else {
        const act = (action) => () => { subjectUi.name = sub.subject; subjectUi.action = action; subjectUi.error = ''; renderSubjects(); };
        li.append(info, h('div', { class: 'subject__actions' },
          sub.banned
            ? h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Lift ban', 'aria-label': `Lift the ban on ${sub.subject}`, on: { click: act('unban') } })
            : h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Check in', 'aria-label': `Check in ${sub.subject}`, on: { click: () => { if (staffEnroller) { staffEnroller.setName(sub.subject); staffEnroller.focus(); } } } }),
          h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Rename', 'aria-label': `Rename ${sub.subject}`, on: { click: act('rename') } }),
          h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: sub.banned ? 'Edit ban' : 'Ban', 'aria-label': `${sub.banned ? 'Edit the ban on' : 'Ban'} ${sub.subject}`, on: { click: act('ban') } }),
          h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Remove', 'aria-label': `Remove ${sub.subject}`, on: { click: act('delete') } })));
      }
      if (active && subjectUi.error) li.append(h('p', { class: 'statusline statusline--error', text: subjectUi.error }));
      return li;
    }));
    if (focusEl) { focusEl.focus(); if (focusEl.select) focusEl.select(); }
    else if (restore) focusSubjectAction(restore.name, restore.action);
  }
  function focusSubjectAction(name, action) {
    const label = `${action === 'delete' ? 'Remove' : 'Rename'} ${name}`;
    $$('#subject-list .subject__actions button').find((button) => button.getAttribute('aria-label') === label)?.focus();
  }

  // ------------------------------------------------------------------ //
  // WebRTC helpers shared by host and guest
  // ------------------------------------------------------------------ //
  const CONN_TEXT = {
    idle: 'Not connected', waiting: 'Waiting for participant', new: 'Connecting…', connecting: 'Connecting…',
    connected: 'Connected', disconnected: 'Disconnected — trying to recover', failed: 'Connection failed', closed: 'Call ended',
  };
  function setConn(el, state, text) { el.dataset.state = state; el.textContent = text || CONN_TEXT[state] || state; }
  const sdpOf = (msg) => (typeof msg.sdp === 'string' ? msg.sdp : msg.sdp && msg.sdp.sdp) || '';
  function closePeer(pc) {
    if (!pc) return;
    pc.onicecandidate = pc.ontrack = pc.onconnectionstatechange = pc.oniceconnectionstatechange = null;
    try { pc.close(); } catch (e) { /* already closed */ }
  }
  const peerState = (pc) => (pc.connectionState && pc.connectionState !== 'new' ? pc.connectionState
    : { checking: 'connecting', completed: 'connected' }[pc.iceConnectionState] || pc.iceConnectionState || 'new');

  // ------------------------------------------------------------------ //
  // Audience call room (host side)
  // ------------------------------------------------------------------ //
  const room = { info: null, ws: null, pc: null, want: false, attempt: 0, timer: null, queue: [], fullRetries: 0, restartTimer: null };
  let remoteStage = null, remotePanel = null, remoteColumn = null;

  function roomDetail(text) { $('#room-detail').textContent = text || ''; }
  function setRemoteStream(stream) {
    capVideo[1].srcObject = stream || null;
    if (stream) capVideo[1].play().catch(() => {});
    remoteStage.setStream(stream, 'Waiting for the participant’s video');
    if (!stream) { S.last[1] = null; remotePanel.idle('No participant video'); }
    else remotePanel.idle('Waiting for the first result');
    emit('remote');
    ensureLink(); pump(1);
  }
  function roomSignal(msg) {
    if (room.ws && room.ws.readyState === 1) { room.ws.send(JSON.stringify(msg)); return true; }
    return false;
  }
  function dropCall() {
    clearTimeout(room.restartTimer);
    closePeer(room.pc); room.pc = null; room.queue = [];
    setRemoteStream(null);
  }
  function connectRoomSignal() {
    if (!room.info || room.ws) return;
    clearTimeout(room.timer);
    const ws = new WebSocket(wsUrl(`/ws/room/${encodeURIComponent(room.info.room_id)}?role=host&token=${encodeURIComponent(room.info.host_token || '')}`));
    room.ws = ws;
    ws.addEventListener('message', (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      onRoomMessage(msg).catch((err) => roomDetail(`Call setup problem: ${err.message}`));
    });
    ws.addEventListener('close', () => {
      if (room.ws !== ws) return;
      room.ws = null;
      if (!room.want) return;
      const delay = backoff(room.attempt++);
      roomDetail(`Signaling link lost. Reconnecting in ${Math.round(delay / 1000)} s…`);
      room.timer = setTimeout(connectRoomSignal, delay);
    });
  }
  async function onRoomMessage(msg) {
    const conn = $('#room-conn');
    if (msg.type === 'joined') {
      room.attempt = 0; room.fullRetries = 0;
      roomDetail('');
      if (msg.peer_present) await startCall(); else if (!room.pc) setConn(conn, 'waiting');
    } else if (msg.type === 'peer-joined') {
      await startCall();
    } else if (msg.type === 'peer-left' || msg.type === 'bye') {
      dropCall();
      setConn(conn, 'waiting', 'Participant left — waiting for them to rejoin');
      roomDetail('The join link stays valid.');
    } else if (msg.type === 'answer' && room.pc) {
      await room.pc.setRemoteDescription({ type: 'answer', sdp: sdpOf(msg) });
      for (const c of room.queue.splice(0)) await room.pc.addIceCandidate(c).catch(() => {});
      room.pc.refreshState();
    } else if (msg.type === 'candidate' && room.pc && msg.candidate) {
      if (room.pc.remoteDescription) await room.pc.addIceCandidate(msg.candidate).catch(() => {});
      else room.queue.push(msg.candidate);
    } else if (msg.type === 'error') {
      if (msg.code === 'room-not-found') {
        room.want = false; dropCall();
        setConn(conn, 'failed', 'Room expired');
        roomDetail('The server no longer knows this room (it may have restarted). End the call and start a new room.');
      } else if (msg.code === 'room-full') {
        if (room.fullRetries++ >= 3) { room.want = false; setConn(conn, 'failed', 'Room already has a host'); roomDetail('Another dashboard is connected to this room as host.'); }
      } else if (msg.code === 'peer-absent') roomDetail('The participant is not connected right now.');
    }
  }
  async function startCall() {
    dropCall();
    roomSignal({ type: 'bye', reason: 'renegotiate' }); // tells a participant with an old connection to drop it
    const pc = new RTCPeerConnection({ iceServers: room.info.ice_servers || [] });
    room.pc = pc;
    const track = cam.stream && cam.stream.getVideoTracks()[0];
    if (track) pc.addTrack(track, cam.stream); else pc.addTransceiver('video', { direction: 'recvonly' });
    pc.onicecandidate = (ev) => { if (ev.candidate) roomSignal({ type: 'candidate', candidate: ev.candidate.toJSON() }); };
    pc.ontrack = (ev) => { if (room.pc === pc) setRemoteStream(ev.streams[0] || new MediaStream([ev.track])); };
    const update = () => {
      if (room.pc !== pc) return;
      const state = peerState(pc);
      setConn($('#room-conn'), state, state === 'failed' ? 'Connection failed — use Reconnect' : undefined);
      roomDetail(state === 'connected' ? (track ? '' : 'Your camera is off, so the participant cannot see you.') : '');
    };
    pc.onconnectionstatechange = update; pc.oniceconnectionstatechange = update;
    pc.refreshState = update;
    setConn($('#room-conn'), 'connecting');
    await pc.setLocalDescription(await pc.createOffer());
    if (room.pc === pc) roomSignal({ type: 'offer', sdp: pc.localDescription.sdp });
  }
  async function reconnectCall() {
    if (!room.info) return;
    room.want = true; room.fullRetries = 0;
    if (!room.ws) { room.attempt = 0; connectRoomSignal(); roomDetail('Reconnecting the signaling link…'); return; }
    const pc = room.pc;
    if (!pc || pc.signalingState === 'closed' || peerState(pc) === 'failed' || pc.signalingState !== 'stable') {
      roomDetail('Setting the call up again…');
      await startCall().catch((err) => roomDetail(`Reconnect failed: ${err.message}`));
      return;
    }
    roomDetail('Restarting the network path…');
    try {
      await pc.setLocalDescription(await pc.createOffer({ iceRestart: true }));
      roomSignal({ type: 'offer', sdp: pc.localDescription.sdp });
      clearTimeout(room.restartTimer);
      room.restartTimer = setTimeout(() => { if (room.pc === pc && peerState(pc) !== 'connected') startCall().catch(() => {}); }, 6000);
    } catch (err) { await startCall().catch((e) => roomDetail(`Reconnect failed: ${e.message}`)); }
  }
  async function createRoom() {
    const button = $('#room-create');
    button.disabled = true;
    try {
      if (!cam.stream) await startCamera();
      const info = await api('/api/rooms', { method: 'POST', json: { base_url: location.origin } });
      room.info = info; room.want = true; room.attempt = 0; room.fullRetries = 0;
      $('#room-url').value = info.join_url;
      $('#room-qr').src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(info.qr_svg || '');
      fill($('#room-warnings'), (info.warnings || []).map((w) => h('li', { class: 'notice notice--warn', text: w })));
      const lan = info.lan_origins || [];
      $('#room-lan').hidden = lan.length === 0;
      fill($('#room-lan-list'), lan.map((origin) => {
        const url = `${origin}/?room=${encodeURIComponent(info.room_id)}`;
        return h('li', { class: 'copyrow' }, h('a', { href: url, target: '_blank', rel: 'noopener', text: url }),
          h('button', { class: 'btn btn--small', type: 'button', text: 'Copy', on: { click: () => copyText(url) } }));
      }));
      $('#room-panel').hidden = false; button.hidden = true;
      if ($('#room-dialog').open) $('#room-copy').focus();
      remoteColumn.hidden = false; emit('remote');
      setConn($('#room-conn'), 'waiting');
      connectRoomSignal();
    } catch (err) { toast(`The room could not be created: ${err.message}`, 'error'); }
    finally { button.disabled = false; }
  }
  function endRoom() {
    roomSignal({ type: 'bye', reason: 'ended' });
    room.want = false; clearTimeout(room.timer);
    dropCall();
    if (room.ws) { const ws = room.ws; room.ws = null; ws.close(1000); }
    // Revoke the join link on the server so it cannot be reused after the call.
    if (room.info) api('/api/rooms/' + encodeURIComponent(room.info.room_id), { method: 'DELETE' }).catch(() => {});
    room.info = null;
    $('#room-panel').hidden = true; $('#room-create').hidden = false;
    $('#room-qr').removeAttribute('src');
    remoteColumn.hidden = true; emit('remote');
    $('#room-create').focus();
  }
  async function copyText(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        const ta = h('textarea', { class: 'visually-hidden', value: text }); document.body.append(ta); ta.select();
        const ok = document.execCommand('copy'); ta.remove();
        if (!ok) throw new Error('copy command rejected');
      }
      toast('Link copied to the clipboard.', 'ok', 2500);
    } catch (e) { toast('Copy failed. Select the link and copy it manually.', 'error'); }
  }
  function onLocalCameraChanged() {
    const pc = room.pc;
    if (!pc) return;
    const track = cam.stream ? cam.stream.getVideoTracks()[0] : null;
    const sender = pc.getSenders()[0];
    if (sender) sender.replaceTrack(track).catch(() => {});
    else if (track) startCall().catch(() => {});
  }


  // ------------------------------------------------------------------ //
  // Optional server-side camera (USB / RTSP source read by the server)
  // ------------------------------------------------------------------ //
  const srv = { running: false, timer: 0, stage: null, panel: null, column: null, button: null, lastCount: 0, lastAt: 0, fps: 0, busy: false };
  function srvSession() {
    const m = effectiveMode();
    return { mode: m.mode, claimed_subject: m.claimed, compare_baseline: false };
  }
  function srvSet(running, note) {
    srv.running = running;
    clearInterval(srv.timer);
    srv.button.setAttribute('aria-pressed', String(running));
    srv.button.lastChild.textContent = running ? 'Stop server camera' : 'Start server camera';
    srv.column.hidden = !running && !note;
    if (running) {
      // <img> cannot send the key header; use a one-minute, single-use stream token instead.
      api('/api/server-camera/stream-token', { method: 'POST' })
        .then((grant) => { if (srv.running) srv.stage.setStream('/api/server-camera/stream.mjpg?token=' + encodeURIComponent(grant.token)); })
        .catch(() => {});
      srv.panel.idle('Waiting for the first result');
      srv.lastCount = 0; srv.fps = 0;
      srv.timer = setInterval(srvPoll, 250);
    } else {
      srv.stage.setStream(null, note || 'Server camera is stopped');
      srv.panel.idle(note ? 'Server camera unavailable' : 'Server camera is stopped', note || '');
    }
    emit('remote');
  }
  async function srvPoll() {
    if (srv.busy || !srv.running) return;
    srv.busy = true;
    try {
      const r = await apiRaw('/api/server-camera/result');
      if (!srv.running) return;
      if (!r.ok) throw new ApiError(r.status, detailOf(r.data) || `HTTP ${r.status}`);
      const st = r.data.status || {}, res = r.data.result;
      if (!st.running) { srvSet(false, st.error ? `The server camera stopped: ${st.error}` : 'The server camera stopped.'); return; }
      const now = performance.now(), count = Number(st.frames_analysed) || 0;
      if (srv.lastCount && count > srv.lastCount && now > srv.lastAt) srv.fps = ((count - srv.lastCount) * 1000) / (now - srv.lastAt);
      if (count !== srv.lastCount) { srv.lastCount = count; srv.lastAt = now; }
      if (res && res.type === 'result') { srv.stage.show(res); srv.panel.update(res, srv.fps); }
      else if (res && res.type === 'error') { srv.stage.show(null); srv.panel.idle('Frame could not be analysed', res.message || res.code || ''); }
      else if (st.error) srv.panel.idle('Server camera problem', st.error);
    } catch (err) {
      srv.panel.idle('Server camera status unavailable', err.message);
    } finally { srv.busy = false; }
  }
  async function srvToggle() {
    srv.button.disabled = true;
    try {
      if (srv.running) { srvSet(false); await api('/api/server-camera/stop', { method: 'POST' }); }
      else {
        const st = await api('/api/server-camera/start', { method: 'POST', json: srvSession() });
        if (st.running) srvSet(true); else srvSet(false, st.error ? `The server camera could not start: ${st.error}` : 'The server camera did not start.');
      }
    } catch (err) {
      srvSet(false, `Server camera: ${err.message}`);
      toast(`Server camera: ${err.message}`, 'error');
    } finally { srv.button.disabled = false; }
  }
  async function initServerCamera(streams) {
    srv.stage = createStage({ label: 'Server camera', image: true });
    srv.panel = createFacePanel('Server camera — primary face');
    srv.column = h('div', { class: 'stream', hidden: true }, srv.stage.el, srv.panel.el);
    srv.button = h('button', { class: 'btn', type: 'button', 'aria-pressed': 'false', hidden: true, on: { click: srvToggle } }, icon('camera'), h('span', { text: 'Start server camera' }));
    streams.append(srv.column);
    $('#mode-switch').before(srv.button);
    on('mode', () => { if (srv.running) api('/api/server-camera/start', { method: 'POST', json: srvSession() }).catch((err) => toast(`Server camera: ${err.message}`, 'error')); });
    on('config', async () => {
      const r = await apiRaw('/api/server-camera').catch(() => null);
      const ok = !!(r && r.ok && r.data && r.data.configured);
      srv.button.hidden = !ok; // stays hidden unless the server reports a configured source
      if (ok && r.data.running && !srv.running) srvSet(true);
      if (!ok && srv.running) srvSet(false);
    });
  }

  // ------------------------------------------------------------------ //
  // View 1: live camera
  // ------------------------------------------------------------------ //
  let liveStage, livePanel, peopleStage, staffEnroller;
  function initLive() {
    liveStage = createStage({ label: 'Your camera', mirror: true });
    livePanel = createFacePanel('Result');
    remoteStage = createStage({ label: 'Remote participant' });
    remotePanel = createFacePanel('Remote participant');
    peopleStage = createStage({ label: 'Your camera', mirror: true });
    $('#people-stage').append(peopleStage.el);
    localStages.push(liveStage, peopleStage);
    const roomDlg = $('#room-dialog');
    $('#open-room').prepend(icon('link'));
    $('#open-room').addEventListener('click', () => roomDlg.showModal());
    $('#room-close').addEventListener('click', () => roomDlg.close());
    roomDlg.addEventListener('click', (ev) => { if (ev.target === roomDlg) roomDlg.close(); });
    remoteColumn = h('div', { class: 'stream', hidden: true }, remoteStage.el, remotePanel.el);
    const streams = $('#streams');
    streams.append(h('div', { class: 'stream' }, liveStage.el, livePanel.el), remoteColumn);
    on('remote', () => streams.classList.toggle('streams--two', $$('.stream', streams).filter((x) => !x.hidden).length > 1));
    remoteStage.setStream(null, 'Waiting for the participant’s video');
    remotePanel.idle('No participant video');
    initServerCamera(streams);
    staffEnroller = createEnroller({ allowSource: true, checkIn: true });
    $('#live-enroller').append(staffEnroller.el);

    $$('input[name="mode"]').forEach((r) => r.addEventListener('change', () => {
      S.mode = r.value;
      $('#verify-field').hidden = S.mode !== 'verify';
      if (S.mode === 'verify' && !S.claimed) setStatus($('#live-status'), 'Choose who you expect to see.');
      else setStatus($('#live-status'), '');
      sendLinkConfig(); emit('mode');
    }));
    $('#verify-subject').addEventListener('change', (ev) => {
      S.claimed = ev.target.value;
      setStatus($('#live-status'), S.claimed ? '' : 'Choose who you expect to see.');
      sendLinkConfig(); emit('mode');
    });
    $('#subjects-refresh').addEventListener('click', loadSubjects);
    $('#room-create').addEventListener('click', createRoom);
    $('#room-copy').addEventListener('click', () => copyText($('#room-url').value));
    $('#room-url').addEventListener('focus', (ev) => ev.target.select());
    $('#room-reconnect').addEventListener('click', () => reconnectCall());
    $('#room-end').addEventListener('click', endRoom);

    on('subjects', () => {
      renderSubjects();
      fillSubjectSelect($('#verify-subject'), 'Select a person', true);
      if (S.claimed && $('#verify-subject').value !== S.claimed) { S.claimed = ''; sendLinkConfig(); }
    });
    on('result', ({ id, msg, fps }) => {
      if (id === 1) { remoteStage.show(msg); remotePanel.update(msg, fps); return; }
      if (S.tab === 'live') { liveStage.show(msg); livePanel.update(msg, fps); }
      if (S.tab === 'people') peopleStage.show(msg);
    });
    on('stale', (id) => {
      const text = !S.engineReady ? 'Recognition engine is not ready' : 'Waiting for results';
      if (id === 1) { remoteStage.show(null); if (streamActive(1)) remotePanel.idle(text); }
      else { localStages.forEach((st) => st.show(null)); if (cam.stream) livePanel.idle(text); }
    });
  }
  function onCameraChanged() {
    capVideo[0].srcObject = cam.stream || null;
    if (cam.stream) capVideo[0].play().catch(() => {});
    localStages.forEach((st) => st.setStream(cam.stream));
    if (!cam.stream) { S.last[0] = null; livePanel.idle('Camera is off', 'Start the camera to begin.'); }
    else livePanel.idle(S.engineReady ? 'Looking for a face…' : 'System is starting up', S.engineReady ? '' : 'The video works, but faces are not checked yet.');
    $$('.js-camera').forEach((b) => {
      b.disabled = cam.starting;
      b.setAttribute('aria-pressed', String(!!cam.stream));
      b.lastChild.textContent = cam.starting ? 'Starting camera…' : cam.stream ? 'Stop camera' : 'Start camera';
    });
    onLocalCameraChanged();
    ensureLink(); pump(0);
  }
  function initCameraButtons() {
    $$('[data-slot="camera-button"]').forEach((slot) => {
      slot.replaceWith(h('button', { class: 'btn btn--primary js-camera', type: 'button', 'aria-pressed': 'false',
        on: { click: () => (cam.stream ? stopCamera() : startCamera()) } }, icon('camera'), h('span', { text: 'Start camera' })));
    });
  }

  // ------------------------------------------------------------------ //
  // View 2: lookalike & abstention stress test
  // ------------------------------------------------------------------ //
  const stress = { a: '', b: '', stage: null, ui: null, simSeq: 0 };
  const OUTCOME_ROWS = [
    { key: 'MATCH_A', alt: 'correct_match', label: 'Correctly named A', cls: 'o-correct' },
    { key: 'MATCH_B', alt: 'wrong_match', label: 'Wrongly named B', cls: 'o-wrong' },
    { key: 'UNCERTAIN_REVIEW', alt: 'uncertain', label: 'Sent for review', cls: 'o-uncertain' },
    { key: 'NO_MATCH', alt: 'no_match', label: 'No match', cls: 'o-nomatch' },
  ];
  function outcomeBar(title, counts, note) {
    const rows = OUTCOME_ROWS.map((r) => ({ ...r, n: Number(counts[r.key] !== undefined ? counts[r.key] : counts[r.alt]) || 0 }));
    const total = rows.reduce((a, r) => a + r.n, 0) || 1;
    return h('div', { class: 'outcome' },
      h('div', { class: 'outcome__title' }, h('strong', { text: title }), note ? h('span', { class: 'muted small', text: note }) : null),
      h('div', { class: 'outcome__bar', role: 'img', 'aria-label': `${title}: ` + rows.map((r) => `${r.label} ${r.n}`).join(', ') },
        rows.filter((r) => r.n > 0).map((r) => {
          const seg = h('div', { class: 'outcome__seg ' + r.cls, title: `${r.label}: ${r.n} of ${total}` },
            r.n / total > 0.08 ? h('span', { text: String(r.n) }) : null);
          seg.style.flexGrow = String(r.n);
          return seg;
        })),
      h('ul', { class: 'outcome__legend' }, rows.map((r) => h('li', {}, h('span', { class: 'swatch ' + r.cls }),
        h('span', { text: r.label }), h('strong', { text: `${r.n} (${fmtRate(r.n / total)})` })))));
  }
  function reasonList(reasons) {
    const items = Object.entries(reasons || {}).sort((a, b) => b[1] - a[1]);
    if (!items.length) return null;
    return h('p', { class: 'small muted' }, h('span', { class: 'label', text: 'Why it asked for review: ' }),
      items.map(([k, v]) => `${reasonLabel(k)} (${v})`).join(' · '));
  }
  function initStress() {
    stress.stage = createStage({ label: 'Your camera', mirror: true });
    localStages.push(stress.stage);
    $('#stress-stage').append(stress.stage.el);
    const pick = (which) => (name) => { stress[which] = name; $('#stress-' + which).value = name; renderProbe(S.last[0]); };
    $('#stress-enroll-a').append(createEnroller({ defaultName: 'Person A', onDone: pick('a') }).el);
    $('#stress-enroll-b').append(createEnroller({ defaultName: 'Person B', onDone: pick('b') }).el);
    for (const which of ['a', 'b']) $('#stress-' + which).addEventListener('change', (ev) => { stress[which] = ev.target.value; renderProbe(S.last[0]); });
    on('subjects', () => {
      for (const which of ['a', 'b']) {
        const sel = $('#stress-' + which);
        fillSubjectSelect(sel, 'Select a person');
        if (stress[which] && S.subjects.some((x) => x.subject === stress[which])) sel.value = stress[which]; else stress[which] = sel.value;
      }
      renderProbe(S.last[0]);
    });
    const ga = createGauge(), gb = createGauge(), verdict = verdictBlock();
    const la = h('div', { class: 'label' }), lb = h('div', { class: 'label' });
    const gap = h('p', { class: 'readout' }), explain = h('p', { class: 'probe__explain' }), msg = h('p', { class: 'muted' });
    const body = h('div', { class: 'probe__body' }, verdict.el, la, ga.el, lb, gb.el, gap, explain);
    stress.ui = { ga, gb, la, lb, verdict, gap, explain, msg, body };
    $('#stress-probe').append(msg, body);
    renderProbe(null);
    on('result', ({ id, msg: m }) => { if (id === 0 && S.tab === 'stress') { stress.stage.show(m); renderProbe(m); } });
    on('stale', (id) => { if (id === 0) renderProbe(null); });

    const form = $('#sim-form');
    const outs = () => {
      $('#sim-similarity-out').textContent = Number($('#sim-similarity').value).toFixed(2);
      $('#sim-quality-out').textContent = Number($('#sim-quality').value).toFixed(2);
    };
    outs();
    const run = debounce(runSimulation, 200);
    form.addEventListener('input', () => { outs(); run(); });
    form.addEventListener('submit', (ev) => { ev.preventDefault(); runSimulation(); });
  }
  function renderProbe(m) {
    const ui = stress.ui;
    if (!ui) return;
    const idle = (text) => { ui.msg.textContent = text; ui.msg.hidden = false; ui.body.hidden = true; };
    if (!stress.a || !stress.b) return idle('Choose person A and person B to begin.');
    if (stress.a === stress.b) return idle('Person A and person B must be two different people.');
    if (!cam.stream) return idle('Start the camera to begin the test.');
    if (!S.engineReady) return idle('The system is still starting up.');
    if (!m) return idle('Looking for a face…');
    const face = (m.faces || []).find((f) => f.primary) || (m.faces || [])[0];
    if (!face) return idle('No face detected. Step into view of the camera.');
    ui.msg.hidden = true; ui.body.hidden = false;
    const d = face.decision, th = d.thresholds || S.thresholds;
    const score = (name) => { const c = (face.candidates || []).find((x) => x.subject === name); return c ? c.score : null; };
    const sa = score(stress.a), sb = score(stress.b);
    ui.verdict.decision(d, '');
    ui.la.textContent = `Similarity to ${stress.a} (A): ` + (sa === null ? 'not close' : sa.toFixed(2));
    ui.lb.textContent = `Similarity to ${stress.b} (B): ` + (sb === null ? 'not close' : sb.toFixed(2));
    ui.ga.set(th, sa, null, ['A', '']); ui.gb.set(th, sb, null, ['B', '']);
    if (sa !== null && sb !== null) {
      const g = Math.abs(sa - sb), narrow = g < th.margin;
      ui.gap.textContent = `Difference between A and B: ${g.toFixed(2)}` + (narrow ? ` (less than ${th.margin.toFixed(2)}, too close to tell apart)` : ' (clear difference)');
      ui.gap.dataset.level = narrow ? 'low' : 'ok';
    } else { ui.gap.textContent = ''; ui.gap.dataset.level = 'none'; }
    const reasons = d.reasons || [d.reason];
    ui.explain.textContent = reasons.includes('NARROW_CANDIDATE_MARGIN')
      ? 'A and B look too alike right now, so the system does not name anyone and asks a person to review.'
      : d.zone === 'MATCH_CONFIRMED' ? 'One person is a clear, strong match, so the system names them.'
        : d.zone === 'UNCERTAIN_REVIEW' ? `The system asks for review: ${reasonLabel(d.reason).toLowerCase()}.`
          : d.zone === 'SPOOF_DETECTED' ? 'The face may be a photo or a screen, so no result is trusted.'
            : 'The face is not similar to anyone enrolled.';
    ui.explain.dataset.narrow = String(reasons.includes('NARROW_CANDIDATE_MARGIN'));
  }
  async function runSimulation() {
    const seq = ++stress.simSeq, status = $('#sim-status');
    const both = $('input[name="sim-enrolled"]:checked').value === 'both';
    setStatus(status, 'Running…');
    try {
      const data = await api('/api/stress/lookalike', { method: 'POST', json: {
        similarity: Number($('#sim-similarity').value), quality: Number($('#sim-quality').value),
        both_enrolled: both, group: $('#sim-group').value, trials: Number($('#sim-trials').value) } });
      if (seq !== stress.simSeq) return;
      setStatus(status, '');
      renderSimulation(data);
    } catch (err) { if (seq === stress.simSeq) setStatus(status, `Simulation failed: ${err.message}`, 'error'); }
  }
  function renderSimulation(d) {
    const fairFir = d.fair.false_identification_rate, baseFir = d.baseline.false_identification_rate;
    const examples = d.examples || [];
    fill($('#sim-result'),
      h('p', { class: 'small muted', text: `${d.trials} generated trials · skin type ${d.group} · similarity ${num(d.identity_similarity)} · photo quality ${num(d.probe_quality)}` }),
      d.both_enrolled ? null : h('div', { class: 'notice notice--error' },
        h('strong', { text: 'Only the lookalike is enrolled. ' }),
        'With the real person missing there is no one to compare the lookalike against, so a close enough double can be accepted by both systems. Keep the list of enrolled people complete, raise the match threshold, or add a second check.'),
      outcomeBar('Fairview', d.fair.outcomes, `wrong names: ${fmtRate(fairFir)}`),
      reasonList(d.fair.abstention_reasons),
      outcomeBar('Conventional system', d.baseline.outcomes, `wrong names: ${fmtRate(baseFir)}`),
      examples.length ? h('details', { class: 'disclosure' },
        h('summary', { text: 'Example trials' }),
        h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Example trials' }, h('table', { class: 'table' },
          h('thead', {}, h('tr', {}, ['Fairview', 'Reason', 'Best match', 'Next match', 'Lead', 'Quality', 'Conventional A / B'].map((t) => h('th', { scope: 'col', text: t })))),
          h('tbody', {}, examples.map((ex) => {
            const f = ex.fair || {}, bc = ex.baseline_cosine || {};
            const c = (x) => (x ? `${x.subject} (${num(x.score, 2)})` : '–');
            const outcome = f.zone === 'MATCH_CONFIRMED' ? `Named ${f.subject}${f.subject === 'A' ? ' (correct)' : ' (wrong)'}` : zoneLabel(f.zone);
            return h('tr', {}, h('td', {}, zoneChip(f.zone, outcome)), h('td', { text: reasonLabel(f.reason) }), h('td', { text: c(f.top1) }),
              h('td', { text: c(f.top2) }), h('td', { class: 'num', text: num(f.margin, 3) }), h('td', { class: 'num', text: num(f.quality, 2) }),
              h('td', { class: 'num', text: `${num(bc.A, 3)} / ${num(bc.B, 3)}` }));
          }))))) : null);
  }
  const ADMIN_EVENTS = { SUBJECT_ENROLLED: 'Subject enrolled', SUBJECT_REPLACED: 'Subject template replaced', SUBJECT_RENAMED: 'Subject renamed', SUBJECT_DELETED: 'Subject deleted',
    GALLERY_ERASED: 'Gallery erased', THRESHOLDS_CHANGED: 'Thresholds changed', AUDIT_MIRROR_REPAIRED: 'Audit mirror repaired',
    REVIEW_QUEUE_CLEARED: 'Review queue cleared' };
  function zoneChip(zone, text) {
    return h('span', { class: 'chip', data: { zone: ZONES[zone] ? zone : 'NONE' } }, icon(ZONES[zone] ? ZONES[zone].icon : 'minus'), h('span', { text: text || zoneLabel(zone) }));
  }

  // ------------------------------------------------------------------ //
  // View 3: baseline vs fair engine
  // ------------------------------------------------------------------ //
  const ab = { overlay: 'fair', stages: [], cards: null, frames: 0, withFace: 0, disagree: 0, kinds: new Map(), dirty: false };
  function disagreementKind(f, b) {
    if (f.zone === b.zone && (f.subject || null) === (b.subject || null)) return null;
    if (b.zone === 'MATCH_CONFIRMED') {
      if (f.zone === 'UNCERTAIN_REVIEW') return `Conventional named someone; Fairview asked for review (${reasonLabel(f.reason).toLowerCase()})`;
      if (f.zone === 'SPOOF_DETECTED') return 'Conventional named someone; Fairview flagged a possible spoof';
      if (f.zone === 'NO_MATCH') return 'Conventional named someone; Fairview found no match';
      return 'Both named someone, but different people';
    }
    if (f.zone === 'MATCH_CONFIRMED') return 'Fairview confirmed a match; conventional found none';
    if (f.zone === 'UNCERTAIN_REVIEW') return `Conventional found no match; Fairview asked for review (${reasonLabel(f.reason).toLowerCase()})`;
    if (f.zone === 'SPOOF_DETECTED') return 'Conventional found no match; Fairview flagged a possible spoof';
    return 'Other difference';
  }
  function initAb() {
    const fairStage = createStage({ label: 'Your camera', mirror: true, engineTag: 'Fair' });
    const baseStage = createStage({ label: 'Your camera', mirror: true, engineTag: 'Baseline' });
    ab.stages = [fairStage, baseStage];
    localStages.push(fairStage, baseStage);
    const wrapB = h('div', { class: 'ab-stage', hidden: true }, h('div', { class: 'label', text: 'Conventional system' }), baseStage.el);
    const capA = h('div', { class: 'label', text: 'Fairview', hidden: true });
    $('#ab-stages').append(h('div', { class: 'ab-stage' }, capA, fairStage.el), wrapB);
    const vf = verdictBlock(), vb = verdictBlock();
    const nf = h('p', { class: 'small muted' }), nb = h('p', { class: 'small muted' }), agree = h('p', { class: 'readout' });
    $('#ab-verdicts').append(
      h('section', { class: 'card', 'aria-label': 'Fairview result' }, h('h2', { class: 'card__title' }, h('span', { class: 'swatch s-fair' }), 'Fairview'), vf.el, nf),
      h('section', { class: 'card', 'aria-label': 'Conventional system result' }, h('h2', { class: 'card__title' }, h('span', { class: 'swatch s-base' }), 'Conventional system'), vb.el, nb),
      agree);
    ab.cards = { vf, vb, nf, nb, agree };
    const idle = (text) => { vf.set('NONE', '', text, ''); vb.set('NONE', '', text, ''); nf.textContent = nb.textContent = agree.textContent = ''; agree.dataset.level = 'none'; };
    ab.idle = idle; idle('Camera is off');
    $$('input[name="ab-overlay"]').forEach((r) => r.addEventListener('change', () => {
      ab.overlay = r.value;
      wrapB.hidden = ab.overlay !== 'split';
      $('#ab-stages').classList.toggle('ab-stages--split', ab.overlay === 'split');
      capA.textContent = ab.overlay === 'baseline' ? 'Conventional system' : 'Fairview';
      capA.hidden = ab.overlay !== 'split';
      if (S.last[0]) drawAb(S.last[0]);
    }));
    $('#tally-reset').addEventListener('click', () => { ab.frames = ab.withFace = ab.disagree = 0; ab.kinds.clear(); renderTally(); });
    on('result', ({ id, msg }) => { if (id === 0 && S.tab === 'ab') { drawAb(msg); tallyAb(msg); } });
    on('stale', (id) => { if (id === 0) idle(cam.stream ? (S.engineReady ? 'Looking for a face…' : 'System is starting up') : 'Camera is off'); });
    setInterval(() => { if (ab.dirty) renderTally(); }, 300);
    renderTally();
  }
  function drawAb(msg) {
    const [fairStage, baseStage] = ab.stages, c = ab.cards;
    fairStage.show(msg, ab.overlay === 'baseline' ? 'baseline' : 'fair');
    baseStage.show(msg, 'baseline');
    const face = (msg.faces || []).find((f) => f.primary) || (msg.faces || [])[0];
    if (!face) { ab.idle('No face detected'); return; }
    const f = face.decision, b = face.baseline;
    c.vf.decision(f, '');
    c.nf.textContent = `Score ${num(f.top1 && f.top1.score, 2)} · quality ${num(f.quality, 2)} · liveness ${f.liveness === null ? 'off' : num(f.liveness, 2)}`;
    if (!b) { c.vb.set('NONE', '', 'Waiting for a result', ''); c.nb.textContent = ''; c.agree.textContent = ''; return; }
    c.vb.decision(b, '');
    c.nb.textContent = `Score ${num(b.top1 && b.top1.score, 2)} · no quality, liveness or lookalike checks`;
    const kind = disagreementKind(f, b);
    c.agree.textContent = kind ? `They disagree: ${kind}.` : 'Both systems agree on this frame.';
    c.agree.dataset.level = kind ? 'low' : 'ok';
  }
  function tallyAb(msg) {
    ab.frames++; ab.dirty = true;
    const face = (msg.faces || []).find((f) => f.primary) || (msg.faces || [])[0];
    if (!face || !face.baseline) return;
    ab.withFace++;
    const kind = disagreementKind(face.decision, face.baseline);
    if (!kind) return;
    ab.disagree++;
    ab.kinds.set(kind, (ab.kinds.get(kind) || 0) + 1);
  }
  function renderTally() {
    ab.dirty = false;
    const kinds = Array.from(ab.kinds.entries()).sort((a, b) => b[1] - a[1]);
    fill($('#tally'),
      h('dl', { class: 'stats' },
        h('div', {}, h('dt', { text: 'Frames' }), h('dd', { text: String(ab.frames) })),
        h('div', {}, h('dt', { text: 'With a face' }), h('dd', { text: String(ab.withFace) })),
        h('div', {}, h('dt', { text: 'Disagreed' }), h('dd', { text: ab.withFace ? fmtRate(ab.disagree / ab.withFace) : '0' }))),
      kinds.length ? h('ul', { class: 'tally__list' }, kinds.map(([kind, n]) => {
        const bar = h('div', { class: 'tally__bar' }); bar.style.width = pct(n / ab.disagree);
        return h('li', {}, h('div', { class: 'tally__row' }, h('span', { text: kind }), h('strong', { text: String(n) })), h('div', { class: 'tally__track' }, bar));
      })) : h('p', { class: 'muted small', text: ab.frames ? 'No disagreements yet. Try poor lighting, covering part of your face, or holding up a photo.' : 'Start the camera. Each frame is judged by both systems and every disagreement is counted here.' }));
  }

  // ------------------------------------------------------------------ //
  // Charts (inline SVG built with DOM APIs)
  // ------------------------------------------------------------------ //
  const chartObserver = window.ResizeObserver ? new ResizeObserver((entries) => {
    for (const { target, contentRect } of entries) {
      const width = Math.floor(contentRect.width);
      if (width > 0 && Math.abs((target._w || 0) - width) > 2) target._draw(width);
    }
  }) : null;
  function mountChart(host, drawFn) {
    host._draw = (width) => { host._w = width; drawFn(width); };
    if (chartObserver) chartObserver.observe(host);
    else requestAnimationFrame(() => { if (host.clientWidth) host._draw(host.clientWidth); });
    return host;
  }
  function clearCharts(host) {
    if (chartObserver) $$('.chart', host).forEach((chart) => chartObserver.unobserve(chart));
    if (entranceObserver) $$('.chart', host).forEach((chart) => entranceObserver.unobserve(chart));
    tip.hide();
  }
  function yScale(values, log, h0, h1) {
    const finite = values.filter((v) => typeof v === 'number' && isFinite(v));
    const pos = finite.filter((v) => v > 0);
    if (log && pos.length) {
      const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...pos) * 0.5)));
      const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...pos))));
      const a = Math.log10(lo), b = Math.max(a + 1, Math.log10(hi));
      const ticks = []; const step = b - a > 6 ? 2 : 1;
      for (let e = a; e <= b + 1e-9; e += step) ticks.push(Math.pow(10, e));
      return { log: true, lo, ticks, y: (v) => (v > 0 ? h0 - ((Math.log10(Math.max(v, lo)) - a) / (b - a)) * (h0 - h1) : h0) };
    }
    let max = Math.max(...finite, 0);
    if (max <= 0) max = 1;
    const mag = Math.pow(10, Math.floor(Math.log10(max))), n = max / mag;
    const top = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
    const div = n <= 2 ? 4 : 5;
    return { log: false, lo: 0, ticks: Array.from({ length: div + 1 }, (_, i) => (i / div) * top), y: (v) => h0 - (Math.max(0, v) / top) * (h0 - h1) };
  }
  function legend(series) {
    return h('ul', { class: 'legend' }, series.map(({ cls, name }) => h('li', {},
      s('svg', { class: 'swatch--line', viewBox: '0 0 23 12', 'aria-hidden': 'true' },
        s('path', { class: cls, d: 'M0 6 H23' })), h('span', { text: name }))));
  }
  /* Lines over a shared threshold axis, with a crosshair tooltip. */
  function lineChart({ title, x, series, log = false, markers = [], xLabel }) {
    const host = h('div', { class: 'chart' });
    const table = h('details', { class: 'disclosure fv-values' }, h('summary', { text: 'View chart values' }));
    table.addEventListener('toggle', () => {
      if (table.open && table.children.length === 1) table.append(fairValueTable(title, 'Threshold', x.map((value) => String(value)), series));
    });
    const wrap = h('figure', { class: 'figure' }, h('figcaption', {}, h('span', { class: 'figure__title', text: title }), log ? h('span', { class: 'muted small', text: 'log scale' }) : null), host, table);
    mountChart(host, (W) => {
      const H = 200, m = { l: 56, r: 20, t: 16, b: 34 };
      const sc = yScale(series.flatMap((v) => v.values).concat(log ? [1e-5] : []), log, H - m.b, m.t);
      const x0 = x[0], x1 = x[x.length - 1], px = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
      const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': title });
      for (const t of sc.ticks) {
        const y = sc.y(t);
        svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y, y2: y }), s('text', { class: 'axis', x: m.l - 6, y: y + 3.5, 'text-anchor': 'end' }, fmtRate(t)));
      }
      for (let i = 0; i <= 4; i++) {
        const v = x0 + ((x1 - x0) * i) / 4;
        svg.append(s('text', { class: 'axis', x: px(v), y: H - m.b + 14, 'text-anchor': 'middle' }, v.toFixed(2)));
      }
      svg.append(s('text', { class: 'axis', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, xLabel || 'threshold'));
      for (const mk of markers) {
        if (typeof mk.x !== 'number' || mk.x < x0 || mk.x > x1) continue;
        svg.append(s('line', { class: 'marker', x1: px(mk.x), x2: px(mk.x), y1: m.t - 4, y2: H - m.b }),
          s('text', { class: 'axis axis--strong', x: px(mk.x) + (mk.left ? -4 : 4), y: m.t + 4, 'text-anchor': mk.left ? 'end' : 'start' }, mk.label));
      }
      for (const ser of series) {
        let d = '', pen = false;
        ser.values.forEach((v, i) => {
          if (typeof v !== 'number' || !isFinite(v) || (sc.log && v <= 0)) { pen = false; return; }
          d += `${pen ? 'L' : 'M'}${px(x[i]).toFixed(1)} ${sc.y(v).toFixed(1)}`; pen = true;
        });
        if (d) svg.append(s('path', { class: 'line ' + ser.cls, d }));
      }
      svg.append(s('line', { class: 'axisline', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
      const cross = s('line', { class: 'crosshair', y1: m.t, y2: H - m.b, visibility: 'hidden' });
      const hit = s('rect', { class: 'hit', x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b });
      hit.addEventListener('mousemove', (ev) => {
        const xv = x0 + ((ev.offsetX - m.l) / (W - m.l - m.r)) * (x1 - x0);
        let idx = 0; x.forEach((v, i) => { if (Math.abs(v - xv) < Math.abs(x[idx] - xv)) idx = i; });
        cross.setAttribute('x1', px(x[idx])); cross.setAttribute('x2', px(x[idx])); cross.setAttribute('visibility', 'visible');
        tip.show(ev, `${xLabel || 'threshold'} ${x[idx].toFixed(2)}`, series.map((ser) => ({ cls: ser.cls, name: ser.name, value: fmtRate(ser.values[idx]) })));
      });
      hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hide(); });
      svg.append(cross, hit);
      fill(host, svg);
    });
    return wrap;
  }

  // ------------------------------------------------------------------ //
  // View 4: fairness audit & threshold simulator
  // ------------------------------------------------------------------ //
  const fair = { sliders: {}, seq: 0, shown: 0, data: null, loaded: false, baselineTouched: false, compare: 'baseline', metric: 'fmr', ctrl: null, rfwLoaded: false };
  const ENGINES = [
    { key: 'fair', name: 'Fairview', note: 'Two combined face models, per-person score calibration, a quality check and the match / review / no-match decision.' },
    { key: 'fair_without_cohort', name: 'Fairview without calibration', note: 'The same decision rules, but without the per-person calibration that evens out scores between groups.' },
    { key: 'baseline', name: 'Conventional system', note: 'One fixed similarity threshold, always picks the closest face, no quality check and no review option.' },
    { key: 'baseline_matched_fmr', name: 'Conventional, same overall error', note: 'The conventional system tuned to make the same overall number of false matches, for a like-for-like comparison.' },
  ];
  const METRICS = [
    { key: 'overall_fmr', name: 'Overall false match rate', fmt: fmtRate, tip: 'How often a comparison accepts the wrong person.' },
    { key: 'overall_fnmr', name: 'Overall false non-match rate', fmt: fmtRate, tip: 'How often a comparison rejects the right person.' },
    { key: 'fmr_gap', name: 'False match rate gap', fmt: fmtRate, tip: 'Difference between the highest and lowest false match rates.' },
    { key: 'fnmr_gap', name: 'False non-match rate gap', fmt: fmtRate, tip: 'Difference between the highest and lowest false non-match rates.' },
    { key: 'equalized_odds_difference', name: 'Equalized odds difference', fmt: fmtRate, tip: 'The larger gap across groups in correct approvals or false matches. Zero is ideal.' },
    { key: 'ser_fmr', name: 'False match rate ratio', ratio: true, tip: 'The highest group rate divided by the lowest. One means equal rates.' },
    { key: 'ser_fnmr', name: 'False non-match rate ratio', ratio: true, tip: 'The highest group rate divided by the lowest. One means equal rates.' },
    { key: 'fairness_discrepancy_rate', name: 'Fairness discrepancy rate', fmt: (v) => num(v, 4), tip: 'One minus the weighted average of the false match and false non-match gaps.' },
    { key: 'false_discovery_rate_gap', name: 'False Discovery Rate gap', fmt: fmtRate, tip: 'Gap across groups in the share of asserted matches that are wrong.' },
    { key: 'genuine_abstention_gap', name: 'Abstention gap (genuine)', fmt: fmtRate, tip: 'Gap across groups in how often genuine comparisons are deferred to review.' },
    { key: 'impostor_abstention_gap', name: 'Abstention gap (impostor)', fmt: fmtRate, tip: 'Gap across groups in how often impostor comparisons are deferred to review.' },
  ];
  /* The two views of section 02 ("By group"). */
  const FAIR_VIEWS = [
    { key: 'fmr', label: 'Wrong person accepted', caption: 'False match rate: how often someone else is accepted as the enrolled person.' },
    { key: 'fnmr', label: 'Right person rejected', caption: 'False non-match rate: how often the enrolled person is turned away. Cases Fairview sends to a person for review are not counted as rejections.' },
  ];
  function metricCell(metric, v) {
    if (v === null || v === undefined) {
      return metric.ratio
        ? h('td', { class: 'num', title: 'Unbounded: the best group had zero errors, so the ratio has no finite value.' }, h('abbr', { title: 'Unbounded: the best group had zero errors, so the ratio has no finite value.', text: '∞' }))
        : h('td', { class: 'num' }, h('abbr', { title: 'Undefined at this operating point (no events to compute it from).', text: 'n/a' }));
    }
    return h('td', { class: 'num', text: metric.ratio ? num(v, 2) + '×' : metric.fmt(v) });
  }
  const fairRatio = (v, d = 2) => (v === null || v === undefined ? '∞' : num(v, d) + '×');
  function fairGroupNoun(d, n) {
    const axis = String(d.group_axis || 'group').toLowerCase();
    const noun = /skin/.test(axis) ? 'skin-type' : axis;
    return noun === 'group' || noun === 'groups' ? `${n} groups` : `${n} ${noun} groups`;
  }
  /* One plain-language sentence with the key result. */
  function fairHeadline(d, n) {
    const f = d.fair.summary.ser_fmr, b = d.baseline.summary.ser_fmr;
    const em = (t, cls) => h('span', { class: 'fv-em' + (cls ? ' ' + cls : ''), text: t });
    const lead = `Across ${fairGroupNoun(d, n)}, `;
    const cmp = b === null || b === undefined
      ? '. The conventional system has no finite ratio because its lowest rate is zero.'
      : ['. The ratio for the conventional system is ', em(fairRatio(b, 1), 'fv-em--base'), '.'];
    if (f === null || f === undefined) return [lead, 'the best-off group gets no false matches at all while others do', cmp];
    if (f < 1.05) return [lead, 'every group gets ', em('about the same'), ' rate of false matches', cmp];
    return [lead, 'the highest false match rate is ', em(fairRatio(f, 1)), ' the lowest', cmp];
  }
  function fairStat(value, caption, sub, tipText) {
    return h('div', { class: 'fv-stat', title: tipText || null },
      h('div', { class: 'fv-stat__value', text: value }),
      h('div', { class: 'fv-stat__cap', text: caption }),
      sub ? h('div', { class: 'fv-stat__sub', text: sub }) : null);
  }
  function fairKey(series) {
    return h('ul', { class: 'fv-key' }, series.map((x) => h('li', {}, h('span', { class: 'fv-key__dot ' + x.cls, 'aria-hidden': 'true' }), h('span', { text: x.name }))));
  }
  function fairValueTable(label, rowLabel, groups, series) {
    const table = h('table', { class: 'table fv-table', 'aria-label': label },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: rowLabel }),
        series.map((ser) => h('th', { scope: 'col', class: 'num', text: ser.name })))),
      h('tbody', {}, groups.map((group, index) => h('tr', {},
        h('th', { scope: 'row', text: group }),
        series.map((ser) => h('td', { class: 'num', text: fmtRate(ser.values[index]) }))))));
    return h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': label }, table);
  }
  function fairValues(label, groups, series) {
    return h('details', { class: 'disclosure fv-values' }, h('summary', { text: 'View chart values' }),
      fairValueTable(label, 'Group', groups, series));
  }
  /* Minimal grouped bar chart: thin bars, faint grid, tooltip per group. */
  function fairBars({ label, groups, series, log = false, height = 240, entrance = false }) {
    const host = h('div', { class: 'chart fv-chart' });
    mountChart(host, (W) => {
      const H = height, m = { l: 52, r: 4, t: 12, b: 28 };
      const sc = yScale(series.flatMap((x) => x.values), log, H - m.b, m.t);
      const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': label + (log ? ' (log scale)' : '') });
      for (const t of sc.ticks) {
        const y = sc.y(t);
        svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y, y2: y }), s('text', { class: 'axis', x: m.l - 8, y: y + 3.5, 'text-anchor': 'end' }, fmtRate(t)));
      }
      const band = (W - m.l - m.r) / groups.length;
      const bw = Math.max(3, Math.min(10, band * 0.13)), gap = Math.max(2, Math.round(bw * 0.5));
      groups.forEach((g, gi) => {
        const cx = m.l + band * gi + band / 2, x0 = cx - (series.length * (bw + gap) - gap) / 2;
        series.forEach((ser, si) => {
          const v = ser.values[gi], x = x0 + si * (bw + gap);
          if (typeof v !== 'number' || !isFinite(v)) return;
          const y = sc.y(v);
          if (v > 0) svg.append(s('rect', { class: 'fv-barrect ' + ser.cls, x, y: Math.min(y, H - m.b - 1.5), width: bw, height: Math.max(1.5, H - m.b - y), rx: 1 }));
          else svg.append(s('circle', { class: 'fv-zero ' + ser.cls, cx: x + bw / 2, cy: H - m.b - 3, r: 1.5 }));
        });
        svg.append(s('text', { class: 'axis axis--x', x: cx, y: H - 8, 'text-anchor': 'middle' }, g));
        const hit = s('rect', { class: 'hit', x: m.l + band * gi, y: m.t, width: band, height: H - m.t - m.b });
        const rows = series.map((ser) => ({ cls: ser.cls, name: ser.name, value: fmtRate(ser.values[gi]) }));
        hit.addEventListener('mousemove', (ev) => tip.show(ev, `Group ${g}`, rows));
        hit.addEventListener('mouseleave', tip.hide);
        svg.append(hit);
      });
      svg.append(s('line', { class: 'axisline', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
      fill(host, svg);
    });
    if (entrance && !fair.chartEntered) onEnter(host, () => {
      if (fair.chartEntered || !host.isConnected) return;
      fair.chartEntered = true;
      $$('.fv-barrect', host).forEach((bar) => animate(bar, [
        { transform: 'scaleY(.05)' }, { transform: 'scaleY(1)' },
      ], entranceMotion));
    });
    return host;
  }
  /* Dumbbell chart for section 02: one row per group, conventional (grey) and
   * Fairview (blue) on a shared rate axis, joined by a line, with the
   * improvement written at the end of the row. Series 0 = Fairview, 1 = conventional. */
  function fairDumbbell({ label, groups, series, log = false, axisName = '' }) {
    const host = h('div', { class: 'chart fv-chart fv-db' });
    const [fs, bs] = series;
    const groupLabel = (g) => (/fitz|skin/i.test(axisName) ? `Type ${g}` : String(g));
    const change = (f, b) => {
      if (typeof f !== 'number' || typeof b !== 'number') return '';
      if (f === 0 && b === 0) return 'both 0%';
      if (f === 0) return 'none at all';
      if (b === 0) return 'conventional: none';
      const r = b / f;
      if (r >= 1.05) return `${r >= 10 ? Math.round(r) : r.toFixed(1)}× fewer`;
      if (r <= 0.95) return `${(1 / r).toFixed(1)}× more`;
      return 'about the same';
    };
    mountChart(host, (W) => {
      const rowH = 38, top = 26, labelW = 64, noteW = W < 520 ? 84 : 112;
      const H = top + rowH * groups.length + 8;
      const x0 = labelW, x1 = Math.max(x0 + 60, W - noteW - 12);
      const sc = yScale([...fs.values, ...bs.values], log, x0, x1);
      const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': label + (log ? ' (log scale)' : '') });
      for (const t of sc.ticks) {
        const x = sc.y(t);
        svg.append(s('line', { class: 'grid', x1: x, x2: x, y1: top - 6, y2: H - 6 }),
          s('text', { class: 'axis', x, y: 12, 'text-anchor': 'middle' }, fmtRate(t)));
      }
      groups.forEach((g, i) => {
        const cy = top + rowH * i + rowH / 2;
        const f = fs.values[i], b = bs.values[i];
        const fx = sc.y(typeof f === 'number' ? f : 0), bx = sc.y(typeof b === 'number' ? b : 0);
        const row = s('g', { class: 'fv-db__row' });
        if (i % 2 === 0) row.append(s('rect', { class: 'fv-db__band', x: 0, y: cy - rowH / 2, width: W, height: rowH }));
        row.append(
          s('text', { class: 'axis axis--x fv-db__label', x: 0, y: cy + 4 }, groupLabel(g)),
          s('line', { class: 'fv-db__link', x1: Math.min(fx, bx), x2: Math.max(fx, bx), y1: cy, y2: cy }),
          s('circle', { class: 'fv-db__dot fv-db__dot--base ' + bs.cls, cx: bx, cy, r: 5 }),
          s('circle', { class: 'fv-db__dot fv-db__dot--fair ' + fs.cls, cx: fx, cy, r: 6, data: { from: String(bx - fx) } }),
          s('text', { class: 'fv-db__note', x: W, y: cy + 4, 'text-anchor': 'end' }, change(f, b)));
        const hit = s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width: W, height: rowH });
        const rows = series.map((ser) => ({ cls: ser.cls, name: ser.name, value: fmtRate(ser.values[i]) }));
        hit.addEventListener('mousemove', (ev) => tip.show(ev, groupLabel(g), rows));
        hit.addEventListener('mouseleave', tip.hide);
        row.append(hit);
        svg.append(row);
      });
      fill(host, svg);
    });
    // Entrance: each blue dot slides in from where the conventional system sits.
    if (!fair.chartEntered) onEnter(host, () => {
      if (fair.chartEntered || !host.isConnected) return;
      fair.chartEntered = true;
      $$('.fv-db__dot--fair', host).forEach((dot, i) => animate(dot, [
        { transform: `translateX(${Number(dot.dataset.from) || 0}px)`, opacity: 0.2 }, { transform: 'translateX(0)', opacity: 1 },
      ], entranceMotion * 1.6, i * 70));
    });
    return host;
  }
  function fairFigure(title, groups, series, opts = {}) {
    return h('figure', { class: 'fv-fig' },
      h('figcaption', { class: 'fv-fig__cap' }, h('span', { text: title }), opts.log ? h('span', { class: 'fv-muted', text: 'log scale' }) : null),
      series.length > 1 ? fairKey(series) : null,
      fairBars({ label: title, groups, series, log: !!opts.log }),
      fairValues(title, groups, series));
  }
  function fairBlock(title, note, ...kids) {
    return h('section', { class: 'fv-block' }, h('h3', { class: 'fv-block__title', text: title }), note ? h('p', { class: 'fv-muted', text: note }) : null, kids);
  }
  function initFairness() {
    const specs = [...THRESHOLD_FIELDS.slice(0, 4), { key: 'baseline_threshold', label: 'Conventional system threshold', hint: 'Only affects the comparison', min: -0.1, max: 0.9, step: 0.005 }];
    const query = debounce(loadFairness, 200);
    for (const spec of specs) {
      const sl = sliderField({ ...spec, value: spec.key === 'baseline_threshold' ? 0.35 : S.thresholds[spec.key], onInput: () => { if (spec.key === 'baseline_threshold') fair.baselineTouched = true; query(); } });
      fair.sliders[spec.key] = sl;
      $('#fair-sliders').append(sl.el);
    }
    fill($('#fair-metric'), FAIR_VIEWS.map((v) => h('button', { type: 'button', class: 'fv-toggle__btn', 'aria-pressed': String(v.key === fair.metric), data: { metric: v.key }, text: v.label,
      on: { click: () => { fair.metric = v.key; if (fair.data) renderFairGroups(fair.data); else syncFairToggle(); } } })));
    const disclose = (btn, panel, labels) => btn.addEventListener('click', () => {
      const open = panel.hidden; panel.hidden = !open;
      btn.setAttribute('aria-expanded', String(open));
      if (labels) btn.textContent = labels[open ? 1 : 0];
    });
    disclose($('#fair-about'), $('#fair-disclaimer'));
    disclose($('#fair-adjust'), $('#fair-controls'), ['Adjust settings', 'Hide settings']);
    $('#fair-controls').addEventListener('submit', (ev) => ev.preventDefault());
    $('#fair-log').addEventListener('change', () => { if (fair.data) renderFairness(fair.data); });
    $('#fair-reset').addEventListener('click', () => {
      for (const f of THRESHOLD_FIELDS.slice(0, 4)) fair.sliders[f.key].set(S.thresholds[f.key]);
      fair.baselineTouched = false; loadFairness();
    });
    $('#fair-apply').addEventListener('click', async () => {
      const body = {}; for (const f of THRESHOLD_FIELDS.slice(0, 4)) body[f.key] = fair.sliders[f.key].get();
      try {
        const r = await api('/api/config/thresholds', { method: 'PUT', json: body });
        S.thresholds = r.thresholds; emit('thresholds');
        toast('Settings applied to the live system.', 'ok');
      } catch (err) { setStatus($('#fair-status'), `Not applied: ${err.message}`, 'error'); }
    });
    on('thresholds', () => { if (!fair.loaded) for (const f of THRESHOLD_FIELDS.slice(0, 4)) fair.sliders[f.key].set(S.thresholds[f.key]); });
  }
  async function loadFairness() {
    fair.loaded = true;
    const seq = ++fair.seq, status = $('#fair-status');
    if (fair.ctrl) fair.ctrl.abort();
    fair.ctrl = new AbortController();
    const q = new URLSearchParams();
    for (const f of THRESHOLD_FIELDS.slice(0, 4)) q.set(f.key, String(fair.sliders[f.key].get()));
    if (fair.baselineTouched) q.set('baseline_threshold', String(fair.sliders.baseline_threshold.get()));
    setStatus(status, fair.data ? 'Updating…' : 'Preparing the report…');
    try {
      const data = await api('/api/fairness/audit?' + q, { signal: fair.ctrl.signal });
      if (seq < fair.shown) return; // an out-of-order reply for an older slider position
      fair.shown = seq; fair.data = data;
      if (seq === fair.seq) setStatus(status, '');
      if (!fair.baselineTouched && typeof data.baseline_cosine_threshold === 'number') fair.sliders.baseline_threshold.set(Number(data.baseline_cosine_threshold.toFixed(3)));
      renderFairness(data);
    } catch (err) {
      if (err.name === 'AbortError' || seq !== fair.seq) return;
      setStatus(status, err.status === 422 ? `These settings are not valid: ${err.message}` : `The report could not be created: ${err.message}`, 'error');
    }
  }
  function renderFairness(d) {
    const groups = d.groups || Object.keys(d.fair.groups);
    $('#fair-kind').textContent = d.kind === 'synthetic_simulation' ? 'Based on simulated data, not measurements of real people.' : 'Based on the evaluation data set.';
    $('#fair-disclaimer').textContent = d.disclaimer || '';
    $('#fair-about').hidden = !d.disclaimer;
    renderFairSummary(d, groups);
    renderFairGroups(d);
    const th = d.thresholds || {};
    $('#fair-current').textContent = `Match at ${num(th.tau_high)} or above · no match below ${num(th.tau_low)} · lookalike margin ${num(th.margin)} · minimum quality ${num(th.q_min)}`;
    renderFairDetails(d, groups);
  }
  function renderFairSummary(d, groups) {
    const fs = d.fair.summary, bs = d.baseline.summary;
    const conv = (v) => `Conventional: ${v}`;
    const summary = $('#fair-summary');
    if (entranceObserver) $$('.fv-stat__value', summary).forEach((el) => entranceObserver.unobserve(el));
    fill(summary,
      h('p', { class: 'fv-headline' }, fairHeadline(d, groups.length)),
      h('div', { class: 'fv-stats' },
        fairStat(fairRatio(fs.ser_fmr), 'Worst vs. best group, false matches', conv(fairRatio(bs.ser_fmr)), METRICS[5].tip),
        fairStat(fmtRate(fs.fmr_gap), 'Largest gap between groups in false matches', conv(fmtRate(bs.fmr_gap)), METRICS[2].tip),
        fairStat(fmtRate(fs.fnmr_gap), 'Largest gap between groups in wrongful rejections', conv(fmtRate(bs.fnmr_gap)), METRICS[3].tip)),
      fs.genuine_abstention_gap > 0 ? h('p', { class: 'fv-muted fv-foot' },
        `Fairview sends unclear cases to a person instead of guessing. How often that happens differs by up to ${fmtRate(fs.genuine_abstention_gap)} between groups.`) : null);
    const values = $$('.fv-stat__value', summary);
    countOnEnter('fair-ratio', values[0], fs.ser_fmr, fairRatio);
    countOnEnter('fair-false-matches', values[1], fs.fmr_gap, fmtRate);
    countOnEnter('fair-false-rejections', values[2], fs.fnmr_gap, fmtRate);
  }
  function syncFairToggle() {
    for (const b of $$('#fair-metric .fv-toggle__btn')) b.setAttribute('aria-pressed', String(b.dataset.metric === fair.metric));
  }
  function renderFairGroups(d) {
    syncFairToggle();
    const groups = d.groups || Object.keys(d.fair.groups), view = FAIR_VIEWS.find((v) => v.key === fair.metric) || FAIR_VIEWS[0];
    const log = $('#fair-log').checked;
    const col = (engine) => groups.map((g) => (d[engine].groups[g] || {})[view.key]);
    const series = [{ name: 'Fairview', cls: 's-fair', values: col('fair') }, { name: 'Conventional system', cls: 's-base', values: col('baseline') }];
    const host = $('#fair-groups');
    const label = `${view.label} by ${d.group_axis || 'group'}`;
    clearCharts(host);
    fill(host,
      fairKey(series),
      fairDumbbell({ label, groups, series, log, axisName: d.group_axis || '' }),
      fairValues(label, groups, series),
      h('p', { class: 'fv-muted', text: `${view.caption} Groups: ${d.group_axis || 'group'}.` }));
  }
  function renderFairDetails(d, groups) {
    const log = $('#fair-log').checked, engines = ENGINES.filter((e) => d[e.key]);
    const col = (engine, field) => groups.map((g) => (d[engine].groups[g] || {})[field]);
    const curves = d.curves || {}, th = d.thresholds || {};
    const groupSeries = (engine, field) => groups.map((g, i) => ({ name: `Group ${g}`, cls: 'g-' + (i + 1), values: ((curves[engine].groups[g] || {})[field]) || [] }));
    const fairMarks = [{ x: th.tau_low, label: 'no match', left: true }, { x: th.tau_high, label: 'match' }];
    const baseMarks = [{ x: d.baseline_cosine_threshold, label: 'threshold' }];
    const look = d.lookalikes || {};
    const lookBlock = (key, heading, text) => (look[key] ? h('div', { class: 'fv-look' }, h('h4', { class: 'fv-look__title', text: heading }), h('p', { class: 'fv-muted', text }),
      outcomeBar('Fairview', look[key].fair, `wrong names: ${fmtRate(look[key].fair.wrong_match_rate)}`), reasonList(look[key].fair.abstention_reasons),
      outcomeBar('Conventional system', look[key].baseline, `wrong names: ${fmtRate(look[key].baseline.wrong_match_rate)}`)) : null);
    const cal = d.calibration || {};
    const ENGINE_CLS = { fair: 's-fair', fair_without_cohort: 's-alt1', baseline: 's-base', baseline_matched_fmr: 's-alt2' };
    const allSeries = (field) => engines.map((e) => ({ name: e.name, cls: ENGINE_CLS[e.key], values: col(e.key, field) }));

    clearCharts($('#fair-body'));
    fill($('#fair-body'),
      fairBlock('How often Fairview asks for review', 'Instead of guessing on unclear images or close calls, Fairview can send a case to a person.',
        fairFigure('Share of comparisons sent to review', groups, [
          { name: 'Same person', cls: 's-fair', values: col('fair', 'genuine_abstention') },
          { name: 'Different people', cls: 's-alt1', values: col('fair', 'impostor_abstention') }])),
      fairBlock('All systems compared', 'Also Fairview without its calibration step, and the conventional system tuned to the same overall error.',
        h('div', { class: 'fv-grid2' },
          fairFigure('Wrong person accepted', groups, allSeries('fmr'), { log }),
          fairFigure('Right person rejected', groups, allSeries('fnmr')))),
      fairBlock('All fairness metrics', null,
        h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Fairness summary' }, h('table', { class: 'table fv-table' },
          h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'Metric' }), engines.map((e) => h('th', { scope: 'col', class: 'num', text: e.name })))),
          h('tbody', {}, METRICS.map((mt) => h('tr', {}, h('th', { scope: 'row' }, h('span', { text: mt.name }), h('small', { class: 'fv-metric-note', text: mt.tip })),
            engines.map((e) => metricCell(mt, d[e.key].summary[mt.key]))))))),
        h('dl', { class: 'fv-notes' }, engines.map((e) => [h('dt', { text: e.name }),
          h('dd', { text: e.note + (e.key === 'baseline_matched_fmr' && typeof d[e.key].cosine_threshold === 'number' ? ` Cosine threshold ${d[e.key].cosine_threshold.toFixed(3)}.` : '') })]))),
      fairBlock('Per-group values', null,
        h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Per-group rates' }, h('table', { class: 'table fv-table' },
          h('thead', {}, h('tr', {}, ['System', 'Group', 'False match rate', 'False non-match rate', 'Correct approval rate', 'Same-person review', 'Different-person review', 'False discovery rate'].map((t, i) => h('th', { scope: 'col', class: i > 1 ? 'num' : '', text: t })))),
          h('tbody', {}, engines.flatMap((e) => groups.map((g) => {
            const r = d[e.key].groups[g] || {};
            return h('tr', {}, h('td', { text: e.name }), h('td', { text: g }), ['fmr', 'fnmr', 'tpr', 'genuine_abstention', 'impostor_abstention', 'false_discovery_rate'].map((k) => h('td', { class: 'num', text: fmtRate(r[k]) })));
          })))))),
      curves.fair && curves.baseline ? fairBlock('Error rate at every threshold', 'When lines overlap, the groups have similar error rates at that setting.',
        legend(groups.map((g, i) => ({ name: `Group ${g}`, cls: 'g-' + (i + 1) }))),
        h('div', { class: 'fv-grid2' },
          lineChart({ title: 'Fairview: wrong person accepted', x: curves.fair.threshold, series: groupSeries('fair', 'fmr'), log, markers: fairMarks, xLabel: 'match threshold' }),
          lineChart({ title: 'Conventional: wrong person accepted', x: curves.baseline.threshold, series: groupSeries('baseline', 'fmr'), log, markers: baseMarks, xLabel: 'similarity threshold' }),
          lineChart({ title: 'Fairview: right person rejected', x: curves.fair.threshold, series: groupSeries('fair', 'fnmr'), markers: fairMarks, xLabel: 'match threshold' }),
          lineChart({ title: 'Conventional: right person rejected', x: curves.baseline.threshold, series: groupSeries('baseline', 'fnmr'), markers: baseMarks, xLabel: 'similarity threshold' }))) : null,
      look.both_enrolled || look.lookalike_only ? fairBlock('Lookalike results', 'Generated pairs where person B looks almost exactly like person A. Every probe is person A.',
        h('div', { class: 'fv-grid2' },
          lookBlock('both_enrolled', 'Both A and B enrolled', 'The system can tell them apart, or ask for review.'),
          lookBlock('lookalike_only', 'Only the lookalike B enrolled', 'With A missing there is nobody to compare against, so both systems can be fooled.'))) : null,
      Object.keys(cal).length ? fairBlock('Calibration used for this report', null,
        h('dl', { class: 'kv' }, Object.entries(cal).map(([k, v]) => [h('dt', { text: humanize(k) }), h('dd', { text: typeof v === 'object' ? JSON.stringify(v) : String(v) })]))) : null);
  }
  async function loadRfw(targetFmr) {
    const section = $('#rfw');
    try {
      const d = await api('/api/fairness/rfw' + (targetFmr ? `?target_fmr=${targetFmr}` : ''));
      fair.rfwLoaded = true;
      if (!d || !d.available) { section.hidden = true; return; }
      section.hidden = false;
      renderRfw(d);
    } catch (err) {
      if (!section.hidden) toast(`RFW report failed: ${err.message}`, 'error');
    }
  }
  function renderRfw(d) {
    const groups = Object.keys(d.groups || {}), oracle = d.oracle_per_group_thresholds || {}, stats = d.score_statistics || {}, sm = d.summary || {};
    const target = h('select', { id: 'rfw-target', on: { change: (ev) => loadRfw(ev.target.value) } },
      [0.001, 0.005, 0.01, 0.05, 0.1].map((v) => h('option', { value: String(v), text: fmtRate(v), selected: Math.abs(v - d.target_fmr) < 1e-9 })));
    const ser = (name, cls, fn) => ({ name, cls, values: groups.map(fn) });
    const restoreFocus = document.activeElement?.id === 'rfw-target';
    clearCharts($('#rfw-body'));
    fill($('#rfw-body'),
      h('p', { class: 'fv-muted' }, h('strong', { class: 'fv-strong', text: 'Real data, conventional single-threshold system. ' }),
        `${d.dataset || ''}. ${d.note || ''} Groups: ${d.group_axis || 'group'}.`),
      h('div', { class: 'fv-bar' }, h('label', { class: 'fv-select', for: 'rfw-target' }, h('span', { text: 'Target overall false match rate' }), target)),
      h('div', { class: 'fv-stats fv-stats--4' },
        fairStat(Number(d.pairs).toLocaleString('en-US'), 'Pairs evaluated'),
        fairStat(num(d.cosine_threshold, 3), 'Global similarity threshold', `False matches ${fmtRate(sm.overall_fmr)} · false non-matches ${fmtRate(sm.overall_fnmr)}`),
        fairStat(fmtRate(sm.fmr_gap), 'Gap in false matches', `Worst vs. best ${fairRatio(sm.ser_fmr)}`),
        fairStat(fmtRate(sm.fnmr_gap), 'Gap in wrongful rejections', `Equalized Odds difference ${fmtRate(sm.equalized_odds_difference)}`)),
      fairBlock('One threshold vs. a threshold per group', 'The per-group (oracle) thresholds require knowing each person’s group label, which a deployed system should not rely on; they show how far a single global threshold is from equal treatment.',
        h('div', { class: 'fv-grid2' },
          fairFigure('Wrong person accepted', groups, [
            ser('One global threshold', 's-base', (g) => d.groups[g].fmr), ser('Per-group thresholds', 's-fair', (g) => (oracle[g] || {}).fmr)]),
          fairFigure('Right person rejected', groups, [
            ser('One global threshold', 's-base', (g) => d.groups[g].fnmr), ser('Per-group thresholds', 's-fair', (g) => (oracle[g] || {}).fnmr)]))),
      fairBlock('Per-group values', null,
        h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'RFW per-group results' }, h('table', { class: 'table fv-table' },
          h('thead', {}, h('tr', {}, ['Group', 'Same-person pairs', 'Different-person pairs', 'False match rate', 'False non-match rate', 'Group threshold', 'Group false match rate', 'Group false non-match rate', 'Mean different-person score', 'Mean same-person score']
            .map((t, i) => h('th', { scope: 'col', class: i ? 'num' : '', text: t })))),
          h('tbody', {}, groups.map((g) => {
            const r = d.groups[g], o = oracle[g] || {}, st = stats[g] || {};
            return h('tr', {}, h('th', { scope: 'row', text: g }), [String(r.n_genuine), String(r.n_impostor), fmtRate(r.fmr), fmtRate(r.fnmr), num(o.cosine_threshold, 3), fmtRate(o.fmr), fmtRate(o.fnmr),
              num(st.impostor_cosine_mean, 3), num(st.genuine_cosine_mean, 3)].map((t) => h('td', { class: 'num', text: t })));
          }))))));
    if (restoreFocus) target.focus({ preventScroll: true });
  }

  // ------------------------------------------------------------------ //
  // View 5: audit log
  // ------------------------------------------------------------------ //
  async function loadAudit() {
    const status = $('#audit-status'), limit = Number($('#audit-limit').value) || 50;
    setStatus(status, 'Loading…');
    try {
      const data = await api(`/api/audit?limit=${limit}`);
      const entries = Array.isArray(data.entries) ? data.entries : [];
      const total = typeof data.total === 'number' ? data.total : null;
      renderAudit(entries);
      setStatus(status, `Showing ${entries.length} of ${total === null ? entries.length : total} entries`);
    } catch (err) { setStatus(status, `The activity log could not be loaded: ${err.message}`, 'error'); }
  }
  function renderAudit(entries) {
    const body = $('#audit-table tbody');
    if (!entries.length) {
      fill(body, h('tr', {}, h('td', { colspan: 7 }, h('div', { class: 'empty audit-empty' },
        h('img', { class: 'empty__image', src: 'assets/img/glass-detail.webp', width: 560, height: 240, loading: 'lazy', decoding: 'async', alt: '' }),
        h('p', { class: 'empty__title', text: 'Nothing has been recorded yet.' })))));
      return;
    }
    // The person a row is about gets a column of their own. A name appears only
    // when the system or a reviewer confirmed it; someone on the banned list is
    // shown in red with a warning mark.
    const personCell = (name) => h('td', {}, !name ? '–' : isBanned(name)
      ? h('span', { class: 'person person--banned', title: 'Banned from premises' }, icon('alert'), h('span', { text: name }))
      : h('span', { class: 'person', text: name }));
    fill(body, entries.map((e) => {
      const t = new Date(e.timestamp);
      const who = e.mode !== 'admin' && e.decision_zone === 'MATCH_CONFIRMED' && e.top1_subject ? e.top1_subject : null;
      if (e.mode === 'review') {
        const key = { REVIEW_SAME_PERSON: 'same_person', REVIEW_DIFFERENT_PERSON: 'different_person', REVIEW_SPOOF_CONFIRMED: 'spoof' }[e.decision_zone];
        const o = REVIEW_OUTCOMES[key] || { label: 'Reviewed', zone: 'NONE', icon: 'minus' };
        const ref = String(e.abstention_reason || '').replace('case:', '');
        return h('tr', { title: `Entry #${e.seq}${e.hash ? ' · ' + e.hash : ''}` },
          h('td', { text: isNaN(t) ? String(e.timestamp || '') : t.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' }) }),
          h('td', {}, h('span', { class: 'chip', data: { zone: 'ADMIN' } }, icon('check'), h('span', { text: 'Human review' }))),
          personCell(key === 'same_person' ? e.top1_subject : null),
          h('td', { class: 'wrapcell', text: `${key === 'same_person' ? 'Confirmed by a reviewer' : o.label} · case #${ref}` }),
          h('td', { class: 'num', text: '–' }), h('td', { class: 'num', text: '–' }), h('td', { class: 'num', text: '–' }));
      }
      const details = e.mode === 'admin'
        ? (e.claimed_subject ? `Was ${e.claimed_subject}` : '–')
        : [e.abstention_reason ? reasonLabel(e.abstention_reason) : null, e.mode === 'verify' && e.claimed_subject ? `Expected ${e.claimed_subject}` : null].filter(Boolean).join(' \u00b7 ') || '\u2013';
      return h('tr', { title: `Entry #${e.seq}${e.hash ? ' \u00b7 ' + e.hash : ''}` },
        h('td', { title: e.timestamp || '', text: isNaN(t) ? String(e.timestamp || '') : t.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' }) }),
        h('td', {}, e.mode === 'admin' ? h('span', { class: 'chip', data: { zone: 'ADMIN' } }, icon('settings'), h('span', { text: ADMIN_EVENTS[e.decision_zone] || humanize(e.decision_zone) }))
          : e.decision_zone === 'NO_FACE_DETECTED' ? zoneChip('NONE', 'No face detected') : zoneChip(e.decision_zone)),
        personCell(e.mode === 'admin' ? e.top1_subject : who),
        h('td', { class: 'wrapcell', text: details }),
        h('td', { class: 'num', text: num(e.top1_score, 2) }),
        h('td', { class: 'num', text: num(e.quality_score, 2) }), h('td', { class: 'num', text: num(e.liveness_score, 2) }));
    }));
  }
  async function verifyAudit() {
    const host = $('#audit-verify-result');
    fill(host, h('p', { class: 'statusline', text: 'Checking the log…' }));
    try {
      const r = await api('/api/audit/verify');
      const errors = r.errors || [];
      fill(host, h('div', { class: 'notice ' + (r.ok ? 'notice--ok' : 'notice--error') },
        h('p', {}, icon(r.ok ? 'check' : 'alert'), h('strong', { text: r.ok ? ' The log is intact. ' : ' The log has been altered. ' }),
          `${r.entries} entr${r.entries === 1 ? 'y' : 'ies'} checked; ${r.ok ? 'no changes were found.' : 'see the errors below.'}`),
        h('details', { class: 'disclosure' }, h('summary', { text: 'Details' }), h('dl', { class: 'kv' },
          h('dt', { text: 'Head hash' }), h('dd', {}, h('code', { class: 'wrap', text: r.head_hash || '–' })),
          h('dt', { text: 'Database chain' }), h('dd', { text: r.sqlite_ok === undefined ? '–' : r.sqlite_ok ? 'intact' : 'broken' }),
          h('dt', { text: 'Append-only file chain' }), h('dd', { text: r.jsonl_ok === undefined ? '–' : r.jsonl_ok ? 'intact' : 'broken' }),
          h('dt', { text: 'Stores consistent' }), h('dd', { text: r.stores_consistent === undefined ? '–' : r.stores_consistent ? 'yes' : 'no' }),
          h('dt', { text: 'Verified at' }), h('dd', { text: r.verified_at || '–' }))),
        errors.length ? h('div', {}, h('p', { class: 'label', text: `Errors (${r.error_count === undefined ? errors.length : r.error_count})` }),
          h('ul', { class: 'plain-list' }, errors.map((e) => h('li', { text: typeof e === 'string' ? e : JSON.stringify(e) })))) : null));
    } catch (err) { fill(host, h('p', { class: 'statusline statusline--error', text: `Verification could not run: ${err.message}` })); }
  }
  async function exportAudit(format) {
    try {
      const blob = await api(`/api/audit/export?format=${format}`, { blob: true });
      const url = URL.createObjectURL(blob);
      const a = h('a', { href: url, download: `audit-log.${format}`, class: 'visually-hidden' });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      toast(`Audit log exported as ${format.toUpperCase()}.`, 'ok', 3000);
    } catch (err) { toast(`Export failed: ${err.message}`, 'error'); }
  }

  // ------------------------------------------------------------------ //
  // Engine status, config, settings drawer
  // ------------------------------------------------------------------ //
  function summarise(v) {
    if (v === null || v === undefined) return '';
    if (Array.isArray(v)) return v.map(summarise).filter(Boolean).join(', ');
    if (typeof v === 'object') {
      if (v.name || v.mode || v.model) return String(v.name || v.mode || v.model) + (v.source ? ` (${v.source})` : '');
      const entries = Object.entries(v);
      if (entries.length && entries.every(([, x]) => x && typeof x === 'object')) return entries.map(([k]) => k).join(', ');
      return entries.filter(([, x]) => typeof x !== 'object').map(([k, x]) => `${humanize(k).toLowerCase()} ${x}`).join(', ');
    }
    return String(v);
  }
  function renderEngine() {
    const pill = $('#engine-pill'), banner = $('#banner');
    const state = !S.serverUp ? 'down' : !S.engineReady ? 'loading' : S.calibrated === false ? 'degraded' : 'ready';
    if (pill.dataset.state !== state) pill.dataset.state = state;
    setText(pill, { ready: 'System ready', degraded: 'Ready \u00b7 not calibrated', down: 'Server offline', loading: 'Starting up\u2026' }[state]);
    banner.hidden = state === 'ready';
    if (state === 'down') setText(banner, 'Can’t reach the server. Trying again…');
    else if (state === 'degraded') setText(banner, 'The system is running without its calibration data, so results are less reliable and groups are not evened out.');
    else if (state === 'loading') setText(banner, S.engineError ? `The recognition system could not start: ${S.engineError}` : 'The recognition system is starting up. Live verification will be available in a moment.');
    updateHomeSystem();
    const engine = (S.config && S.config.engine) || {};
    // Where recognition runs: always this machine, never a cloud service.
    const PROVIDERS = { CUDAExecutionProvider: 'NVIDIA GPU', DmlExecutionProvider: 'GPU (DirectML)', QNNExecutionProvider: 'NPU',
      OpenVINOExecutionProvider: 'Intel AI accelerator', CoreMLExecutionProvider: 'Apple Neural Engine', CPUExecutionProvider: 'CPU' };
    const first = (engine.providers || []).find((p) => PROVIDERS[p]);
    const device = $('#engine-device');
    device.hidden = !(S.engineReady && first);
    if (first) {
      setText(device, `On this device · ${PROVIDERS[first]} · no cloud`);
      device.title = 'Faces are analysed on this computer. No image or face data is sent to an online service.';
    }
    const triState = (v) => (v === true ? 'yes' : v === false ? 'no' : 'unknown / operator supplied');
    const meta = Object.entries(engine).filter(([k, v]) => !['ready', 'error', 'commercial_use_permitted'].includes(k) && summarise(v) !== '');
    const pick = (re) => meta.find(([k]) => re.test(k));
    const head = [pick(/model|embed/i), pick(/provider|device/i), pick(/calib/i)].filter(Boolean);
    fill($('#engine-meta'), (head.length ? head : meta.slice(0, 3)).map(([k, v]) => h('span', { class: 'engine__chip', title: `${humanize(k)}: ${summarise(v)}` },
      h('span', { class: 'muted', text: humanize(k) + ' ' }), summarise(v))));
    const c = S.config || {};
    const models = engine.models && typeof engine.models === 'object' ? Object.entries(engine.models) : [];
    const rows = [...meta.filter(([k]) => k !== 'models').map(([k, v]) => [humanize(k), summarise(v)]),
      ...models.map(([k, m]) => [`Model: ${k}`, [m.file, m.description, m.provider, m.license ? `Licence: ${m.license}` : null, `Commercial use: ${triState(m.commercial_use)}`].filter(Boolean).join(' \u2014 ')]),
      ['Commercial use permitted', triState(engine.commercial_use_permitted)],
      ...(engine.calibration ? [['Calibration detail', [`fitted on data: ${triState(engine.calibration.fitted_on_data)}`, `equalises subgroups: ${triState(engine.calibration.equalises_subgroups)}`,
        `False match rate at the no-match threshold: ${fmtRate(engine.calibration.fmr_at_tau_low_anchor)}`,
        `False match rate at the match threshold: ${fmtRate(engine.calibration.fmr_at_tau_high_anchor)}`].join(' · ')]] : []),
      ['Liveness check', c.liveness_enabled === undefined ? '–' : c.liveness_enabled ? 'enabled' : 'disabled'],
      ['Conventional system similarity threshold', num(c.baseline_cosine_threshold, 3)],
      ['Template storage', c.storage ? `${c.storage.cipher} (${c.storage.key_source}); raw images stored: ${c.storage.raw_images_stored ? 'yes' : 'no'}` : '–'],
      ['Enrollment samples', c.enrollment ? `${c.enrollment.min_samples} to ${c.enrollment.max_samples}, minimum quality ${num(c.enrollment.min_quality)}` : '–']];
    if (!S.engineReady && S.engineError) rows.unshift(['Engine error', S.engineError]);
    fill($('#settings-engine'), rows.map(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })]));
  }
  let healthTimer = 0;
  async function pollHealth() {
    clearTimeout(healthTimer);
    const was = S.engineReady;
    try {
      const r = await apiRaw('/api/health');
      if (!r.ok || !r.data) throw new Error('bad health reply');
      S.serverUp = true; S.engineReady = !!r.data.engine_ready; S.engineError = r.data.engine_error || null;
      S.calibrated = r.data.calibrated === undefined ? null : r.data.calibrated;
    } catch (e) { S.serverUp = false; S.engineReady = false; }
    renderEngine();
    if (S.engineReady && !was) { loadConfig(); pump(0); pump(1); if (cam.stream) livePanel.idle('Waiting for the first result'); }
    if (!S.engineReady && was) emit('stale', 0);
    healthTimer = setTimeout(pollHealth, S.engineReady ? 15000 : 2500);
  }
  async function loadConfig() {
    const r = await apiRaw('/api/config').catch(() => null);
    if (!r) { toast('Configuration could not be loaded: the server could not be reached.', 'error'); return false; }
    if (r.status === 401) return false;
    if (!r.ok) { toast(`Configuration could not be loaded: ${detailOf(r.data) || 'HTTP ' + r.status}`, 'error'); return false; }
    S.config = r.data;
    if (r.data.thresholds) S.thresholds = r.data.thresholds;
    emit('config'); emit('thresholds');
    renderEngine();
    return true;
  }
  function initSettings() {
    const dlg = $('#settings'), sliders = {}, bar = $('.settings__bar');
    // Live preview of the three decision zones, driven by the two thresholds.
    const preview = createGauge();
    $('#settings-preview').append(preview.el);
    const current = () => Object.fromEntries(THRESHOLD_FIELDS.map((f) => [f.key, sliders[f.key].get()]));
    const isDirty = () => THRESHOLD_FIELDS.some((f) => Math.abs(sliders[f.key].get() - S.thresholds[f.key]) > 1e-9);
    const refresh = () => {
      const v = current();
      preview.set({ ...v, tau_low: Math.min(v.tau_low, v.tau_high), tau_high: Math.max(v.tau_low, v.tau_high) }, null, null);
      const dirty = isDirty();
      bar.dataset.dirty = String(dirty);
      $('#settings-apply').disabled = !dirty; $('#settings-revert').disabled = !dirty;
      if (dirty) setStatus($('#settings-status'), 'Unsaved changes');
      else if ($('#settings-status').textContent === 'Unsaved changes') setStatus($('#settings-status'), '');
    };
    const GROUPS = [
      { title: 'Matching', keys: ['tau_high', 'tau_low', 'margin'] },
      { title: 'Safety checks', keys: ['q_min', 'l_min'] },
    ];
    for (const group of GROUPS) {
      const box = h('fieldset', { class: 'settings__group' }, h('legend', { class: 'settings__group-title', text: group.title }));
      for (const key of group.keys) {
        const spec = THRESHOLD_FIELDS.find((f) => f.key === key);
        sliders[key] = sliderField({ ...spec, value: S.thresholds[key], onInput: refresh });
        box.append(sliders[key].el);
      }
      $('#settings-sliders').append(box);
    }
    const sync = () => { for (const f of THRESHOLD_FIELDS) sliders[f.key].set(S.thresholds[f.key]); refresh(); };
    on('thresholds', () => { if (!dlg.open) sync(); });
    $('#open-settings').prepend(icon('settings'));
    $('#open-settings').addEventListener('click', () => { sync(); setStatus($('#settings-status'), ''); dlg.showModal(); });
    $('#settings-close').addEventListener('click', () => dlg.close());
    dlg.addEventListener('click', (ev) => { if (ev.target === dlg) dlg.close(); });
    $('#settings-revert').addEventListener('click', () => { sync(); setStatus($('#settings-status'), 'Changes discarded.'); });
    $('#settings-form').addEventListener('submit', (ev) => ev.preventDefault());
    $('#settings-apply').addEventListener('click', async () => {
      const body = current();
      try {
        const r = await api('/api/config/thresholds', { method: 'PUT', json: body });
        S.thresholds = r.thresholds; sync(); emit('thresholds');
        setStatus($('#settings-status'), 'Saved', 'ok');
      } catch (err) { setStatus($('#settings-status'), `Not saved: ${err.message}`, 'error'); }
    });
    refresh();
    $('#settings-key').addEventListener('click', () => requestKey(''));
    const keyDlg = $('#key-dialog');
    $('#key-close').addEventListener('click', () => keyDlg.close());
    $('#key-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      setKey($('#key-input').value.trim());
      keyDlg.close();
      link.blocked = '';
      if (await loadConfig()) { loadSubjects(); ensureLink(); }
    });
  }

  // ------------------------------------------------------------------ //
  // Tabs
  // ------------------------------------------------------------------ //
  // ------------------------------------------------------------------ //
  // Review queue: a person resolves the cases the system abstained on
  // ------------------------------------------------------------------ //
  const review = { filter: 'open', busy: false, cases: [], open: 0, timer: 0, cards: new Map() };
  const REVIEW_SOURCES = { 'stream:local': 'Your camera', 'stream:remote': 'Remote participant', 'stream:server': 'Server camera' };
  const REVIEW_OUTCOMES = {
    same_person: { label: 'Confirmed as', zone: 'MATCH_CONFIRMED', icon: 'check' },
    different_person: { label: 'Not the same person', zone: 'NO_MATCH', icon: 'cross' },
    spoof: { label: 'Spoof confirmed', zone: 'SPOOF_DETECTED', icon: 'alert' },
  };
  const sourceLabel = (src) => REVIEW_SOURCES[src] || (String(src || '').startsWith('api') ? 'Uploaded image' : 'Camera');
  function timeAgo(iso) {
    const t = new Date(iso).getTime();
    if (!isFinite(t)) return '';
    const s = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (s < 45) return 'just now';
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return new Date(t).toLocaleDateString([], { dateStyle: 'medium' });
  }
  function clockTime(iso) {
    const t = new Date(iso);
    return isFinite(t.getTime()) ? t.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' }) : '';
  }
  function spanLabel(a, b) {
    const s = Math.max(1, Math.round((new Date(b).getTime() - new Date(a).getTime()) / 1000));
    if (!isFinite(s)) return '';
    if (s < 90) return `${s} s`;
    return s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
  }
  function setReviewBadge(n) {
    const badge = $('#review-badge');
    badge.hidden = !n; badge.textContent = n > 99 ? '99+' : String(n || '');
    badge.setAttribute('aria-label', `${n} open`);
    $('#review-open-count').textContent = n ? String(n) : '';
    $('#review-clear').hidden = !n;
  }
  async function clearReview() {
    const n = review.open;
    if (!n || !window.confirm(`Clear all ${n} open case${n === 1 ? '' : 's'}? This is recorded in the activity log.`)) return;
    const btn = $('#review-clear');
    btn.disabled = true;
    try {
      const r = await api('/api/review/clear', { method: 'POST' });
      toast(`Cleared ${r.cleared} case${r.cleared === 1 ? '' : 's'} from the review queue.`, 'ok', 3000);
    } catch (err) {
      setStatus($('#review-status'), `The queue could not be cleared: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
      loadReview(true);
    }
  }
  async function loadReview(quiet) {
    if (review.busy) return;
    review.busy = true;
    if (!quiet) setStatus($('#review-status'), 'Loading…');
    try {
      const data = await api(`/api/review/cases?status=${review.filter}&limit=100`);
      review.cases = data.cases || []; review.open = data.open || 0;
      setReviewBadge(review.open);
      setStatus($('#review-status'), '');
      renderReview();
    } catch (err) { if (!quiet) setStatus($('#review-status'), `The review queue could not be loaded: ${err.message}`, 'error'); }
    finally { review.busy = false; }
  }
  /* Review images need the API key header, which <img src> cannot send: each
   * one is fetched once as a blob and shown through an object URL. */
  const reviewImages = new Map(); // key -> { url, promise }
  function reviewImage(key, path, alt, cls) {
    const img = h('img', { class: cls, alt, decoding: 'async' });
    let entry = reviewImages.get(key);
    if (!entry) {
      entry = { url: null };
      entry.promise = apiRaw(path, { blob: true })
        .then((r) => {
          if (r.ok && r.data && reviewImages.get(key) === entry) entry.url = URL.createObjectURL(r.data);
          return entry.url;
        })
        .catch(() => null);
      reviewImages.set(key, entry);
    }
    entry.promise.then((url) => { if (url) img.src = url; else img.dataset.missing = 'true'; });
    return img;
  }
  function pruneReviewImages(keep) {
    for (const [key, entry] of reviewImages) {
      if (keep.has(key)) continue;
      if (entry.url) URL.revokeObjectURL(entry.url);
      reviewImages.delete(key);
    }
  }
  const frameKey = (seq) => `frame:${seq}`;
  const recordKey = (name) => `record:${name}`;
  const recordPath = (name) => `/api/subjects/${encodeURIComponent(name)}/record-photo`;
  const caseFrames = (c) => (c.frames || []).filter((f) => f.image);
  const caseBanned = (c) => (c.alert === 'banned' && c.banned && c.banned[0]) || null;
  // One case covers a whole visit: show when it started and when the person was last seen.
  const metaText = (c) => [`Case ${c.case_id}`, sourceLabel(c.source),
    c.sightings > 1 ? `${clockTime(c.first_seen)} to ${clockTime(c.last_seen)} (${spanLabel(c.first_seen, c.last_seen)})` : clockTime(c.first_seen),
    `last seen ${timeAgo(c.last_seen)}`].filter(Boolean).join(' · ');
  function caseHeadline(c) {
    const names = (c.candidates || []).map((x) => x.subject);
    // "The image is unclear" is the usual reason a case is here, so it goes
    // unsaid; only a more specific reason is spelled out.
    const why = c.reason && c.reason !== 'LOW_IMAGE_QUALITY' && c.reason !== 'LIVENESS_BELOW_MINIMUM'
      ? `: ${reasonLabel(c.reason).toLowerCase()}` : '';
    const banned = caseBanned(c);
    if (banned) {
      const sure = c.zone === 'MATCH_CONFIRMED';
      return `${sure ? 'Looks like' : 'Might be'} ${banned.subject}, who is banned from the premises` + (sure ? '' : why);
    }
    const who = names.slice(0, 2).join(' or ');
    if (c.zone === 'SPOOF_DETECTED') return `${who ? `Possible spoof of ${who}` : 'Possible spoof, no one close'}${why}`;
    return `${who ? `Might be ${who}` : (c.claimed_subject ? `Claims to be ${c.claimed_subject}` : 'Unknown face')}${why}`;
  }
  /* The clearest camera frames of the case: one large, the rest as thumbnails
   * that swap into the large slot. A banned person's record sits beside them. */
  function caseEvidence(c) {
    const frames = caseFrames(c), banned = caseBanned(c);
    if (c.status === 'resolved') return null; // frames are deleted once a person decides
    const figures = [];
    if (frames.length) {
      const label = (f) => `Frame at ${clockTime(f.timestamp)}, quality ${num(f.quality, 2)}`;
      const lead = reviewImage(frameKey(frames[0].seq), `/api/review/frames/${frames[0].seq}`, 'Face seen at the camera', 'evidence__lead');
      let thumbs = [];
      if (frames.length > 1) {
        thumbs = frames.map((f, i) => {
          const img = reviewImage(frameKey(f.seq), `/api/review/frames/${f.seq}`, '', 'evidence__thumb-img');
          const btn = h('button', { class: 'evidence__thumb', type: 'button', title: label(f), 'aria-label': label(f), 'aria-pressed': String(i === 0), on: { click: () => {
            if (!img.src) return;
            lead.src = img.src;
            thumbs.forEach((t) => t.setAttribute('aria-pressed', String(t === btn)));
          } } }, img);
          return btn;
        });
      }
      figures.push(h('figure', { class: 'evidence__fig' }, lead,
        h('figcaption', { text: frames.length > 1 ? `At the camera · ${frames.length} clearest frames` : 'At the camera' }),
        thumbs.length ? h('div', { class: 'evidence__thumbs' }, thumbs) : null));
    } else {
      figures.push(h('p', { class: 'evidence__none', text: 'No photo for this case: it is older than review photos, or its photos expired. New cases show the clearest frames here.' }));
    }
    if (banned) {
      figures.push(h('figure', { class: 'evidence__fig evidence__fig--record' },
        banned.photo ? reviewImage(recordKey(banned.subject), recordPath(banned.subject), `Record photo of ${banned.subject}`, 'evidence__lead')
          : h('div', { class: 'evidence__lead evidence__lead--empty', text: 'No record photo' }),
        h('figcaption', {}, h('strong', { text: `On record · ${banned.subject}` }), banned.note ? h('span', { text: banned.note }) : null),
        h('p', { class: 'evidence__hint', text: 'From this site’s own banned list. Nothing happens automatically: compare the two and decide.' })));
    }
    return h('div', { class: 'evidence' }, figures);
  }
  function buildCase(c, known) {
    const meta = h('span', { class: 'review-case__meta', text: metaText(c) });
    const names = (c.candidates || []).map((x) => {
      const notes = [c.candidates.length > 1 ? `${x.frames}` : null, x.banned ? 'banned list' : null].filter(Boolean).join(', ');
      return notes ? `${x.subject} (${notes})` : x.subject;
    });
    const facts = h('dl', { class: 'review-case__facts' },
      h('div', {}, h('dt', { text: 'Closest match' }), h('dd', { text: names.join(', ') || 'No one close' })),
      h('div', {}, h('dt', { text: 'Match score' }), h('dd', { text: num(c.top1_score, 2) })),
      h('div', {}, h('dt', { text: 'Image quality' }), h('dd', { text: num(c.quality_score, 2) })),
      h('div', {}, h('dt', { text: 'Liveness' }), h('dd', { text: num(c.liveness_score, 2) })));
    const li = h('li', { class: 'review-case' + (known ? ' review-case--known' : ''), data: { status: c.status, alert: c.alert || '' } },
      h('div', { class: 'review-case__main' },
        h('div', { class: 'review-case__head' },
          caseBanned(c) ? h('span', { class: 'chip', data: { zone: 'SPOOF_DETECTED' } }, icon('alert'), h('span', { text: 'Banned from premises' })) : null,
          caseBanned(c) && c.zone === 'MATCH_CONFIRMED' ? null : zoneChip(c.zone), meta),
        h('p', { class: 'review-case__why', text: caseHeadline(c) }),
        caseEvidence(c),
        facts));
    li.append(c.status === 'resolved' ? reviewedBlock(c) : decisionBlock(c));
    return { li, meta };
  }
  function renderReview() {
    const list = $('#review-list');
    $('#review-empty').hidden = review.cases.length > 0;
    $('#review-empty .empty__title').textContent = review.filter === 'resolved' ? 'No reviewed cases yet' : 'Nothing to review';
    const subjects = S.subjects.map((x) => x.subject).join('\n');
    const cards = new Map(), keep = new Set();
    const items = review.cases.map((c) => {
      if (c.status !== 'resolved') {
        caseFrames(c).forEach((f) => keep.add(frameKey(f.seq)));
        const banned = caseBanned(c);
        if (banned && banned.photo) keep.add(recordKey(banned.subject));
      }
      // A card is rebuilt only when its case changed, so the 6 s refresh does
      // not reset a half-made choice or reload pictures.
      const sig = JSON.stringify([c.status, c.zone, c.reason, c.sightings, c.last_seq, c.candidates, caseFrames(c).map((f) => f.seq),
        c.banned, c.review, c.top1_score, (c.candidates || []).length ? '' : subjects]);
      const old = review.cards.get(c.case_id);
      const card = old && old.sig === sig ? old : { sig, ...buildCase(c, !!old) };
      setText(card.meta, metaText(c));
      cards.set(c.case_id, card);
      return card.li;
    });
    review.cards = cards;
    pruneReviewImages(keep);
    if (items.length !== list.children.length || items.some((li, i) => list.children[i] !== li)) fill(list, items);
  }
  function reviewedBlock(c) {
    const r = c.review || {}, o = REVIEW_OUTCOMES[r.outcome] || { label: 'Reviewed', zone: 'NONE', icon: 'minus' };
    return h('div', { class: 'review-case__done' },
      h('span', { class: 'chip', data: { zone: o.zone } }, icon(o.icon), h('span', { text: r.outcome === 'same_person' && r.subject ? `${o.label} ${r.subject}` : o.label })),
      h('span', { class: 'review-case__meta', text: `Reviewed ${timeAgo(r.timestamp)}` }));
  }
  function decisionBlock(c) {
    const status = h('p', { class: 'statusline', role: 'status', 'aria-live': 'polite' });
    // With several likely people the reviewer picks one; with none, they name the person.
    const likely = (c.candidates || []).map((x) => x.subject);
    const picker = likely.length === 1 ? null : h('select', { 'aria-label': 'Who is it?' },
      likely.length ? likely.map((name) => h('option', { value: name, text: name }))
        : [h('option', { value: '', text: 'Who is it?' }), S.subjects.map((x) => h('option', { value: x.subject, text: x.subject }))]);
    const chosen = () => (picker ? picker.value : likely[0]);
    const buttons = [];
    const decide = (outcome) => async () => {
      const body = { outcome };
      if (outcome === 'same_person') {
        body.subject = chosen();
        if (!body.subject) { setStatus(status, 'Choose who it is first.', 'error'); picker && picker.focus(); return; }
      }
      buttons.forEach((b) => { b.disabled = true; });
      try {
        await api(`/api/review/cases/${c.case_id}`, { method: 'POST', json: body });
        toast('Decision recorded in the activity log.', 'ok', 2500);
        loadReview(true);
      } catch (err) {
        setStatus(status, err.status === 409 ? 'Someone already reviewed this case.' : `Not saved: ${err.message}`, 'error');
        buttons.forEach((b) => { b.disabled = false; });
        if (err.status === 409) loadReview(true);
      }
    };
    const yes = h('button', { class: 'btn btn--primary btn--small', type: 'button', on: { click: decide('same_person') } });
    const nameYes = () => setText(yes, chosen() ? `Yes, it's ${chosen()}` : 'Same person');
    nameYes();
    if (picker) picker.addEventListener('change', nameYes);
    buttons.push(yes,
      h('button', { class: 'btn btn--small', type: 'button', text: likely.length > 1 ? 'Someone else' : 'Different person', on: { click: decide('different_person') } }),
      h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Spoof', title: 'A photo, screen or mask', on: { click: decide('spoof') } }));
    return h('div', { class: 'review-case__actions' }, picker, h('div', { class: 'row' }, buttons), status);
  }
  function initReview() {
    $$('input[name="review-filter"]').forEach((r) => r.addEventListener('change', () => { review.filter = r.value; loadReview(); }));
    $('#review-refresh').addEventListener('click', () => loadReview());
    $('#review-clear').addEventListener('click', clearReview);
    // Keep the nav badge current; refresh the list itself only while it is on screen.
    const tick = async () => {
      if (S.tab === 'review') await loadReview(true);
      else { try { const d = await api('/api/review/cases?status=open&limit=1'); setReviewBadge(d.open || 0); } catch (e) { /* offline */ } }
    };
    review.timer = setInterval(tick, 6000);
    tick();
  }

  const TABS = ['home', 'live', 'review', 'people', 'audit', 'fairness', 'ab', 'stress'];
  const mobileTabs = matchMedia('(max-width: 960px)');
  function revealTab(tab) {
    if (!mobileTabs.matches || !tab) return;
    const list = tab.parentElement;
    const left = tab.offsetLeft, right = left + tab.offsetWidth;
    if (left < list.scrollLeft) list.scrollLeft = left;
    else if (right > list.scrollLeft + list.clientWidth) list.scrollLeft = right - list.clientWidth;
  }
  function updateHomeSystem() {
    const state = !S.serverUp ? 'Offline' : S.engineError ? 'Unavailable' : !S.engineReady ? 'Starting up' : 'Ready';
    setText($('#home-system'), state);
  }
  const formatCount = (n) => Math.round(n).toLocaleString();
  function showHomeCount(id, value) {
    const el = document.getElementById(id);
    if (el.dataset.value === String(value)) return;
    el.dataset.value = value;
    setText(el, Number.isFinite(value) ? formatCount(value) : '–');
    countOnEnter(id, el, value, formatCount);
  }
  async function loadHome() {
    showHomeCount('home-people', S.subjects.length);
    updateHomeSystem();
    try {
      const r = await api('/api/audit?limit=1');
      showHomeCount('home-decisions', r.total);
    } catch (e) { showHomeCount('home-decisions', null); }
  }
  function positionTabIndicator() {
    if (mobileTabs.matches) return;
    const selected = $('.tabs [aria-selected="true"]'), indicator = $('.tabs__indicator');
    if (!selected) return;
    indicator.style.transform = `translateY(${selected.offsetTop + (selected.offsetHeight - 16) / 2}px)`;
    if (!indicator.classList.contains('is-positioned')) {
      requestAnimationFrame(() => indicator.classList.add('is-positioned'));
    }
  }
  function setTab(name, focus) {
    if (!TABS.includes(name)) name = 'home';
    const wasShown = !$('#view-' + name).hidden;
    S.tab = name;
    for (const t of TABS) {
      const btn = $('#tab-' + t), active = t === name;
      $('#view-' + t).hidden = !active;
      if (btn.getAttribute('role') !== 'tab') {
        // The activity log is a quiet link in the sidebar footer, not a main view.
        if (active) btn.setAttribute('aria-current', 'page'); else btn.removeAttribute('aria-current');
        continue;
      }
      btn.setAttribute('aria-selected', String(active)); btn.tabIndex = active ? 0 : -1;
      if (active && focus) btn.focus({ preventScroll: true });
      if (active) revealTab(btn);
    }
    $('.tabs__indicator').classList.toggle('is-hidden', !$('.tabs [aria-selected="true"]'));
    positionTabIndicator();
    tip.hide();
    if (!wasShown) window.scrollTo(0, 0);
    try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ }
    sendLinkConfig();
    localStages.forEach((st) => st.show(null));
    if (name === 'fairness') { if (!fair.loaded) loadFairness(); if (!fair.rfwLoaded) loadRfw(); }
    if (name === 'audit') loadAudit();
    if (name === 'review') loadReview();
    if (name === 'home') loadHome();
    if (name === 'stress' && !stress.simSeq) runSimulation();
    if (name === 'stress') renderProbe(S.last[0]);
  }
  function initTabs() {
    const list = $('.tabs__list');
    const sidebar = $('.sidebar');
    const updateLayout = () => {
      list.setAttribute('aria-orientation', mobileTabs.matches ? 'horizontal' : 'vertical');
      revealTab($('[aria-selected="true"]', list));
      document.documentElement.style.setProperty('--sticky-offset', mobileTabs.matches ? `${sidebar.offsetHeight + 12}px` : '0px');
      positionTabIndicator();
    };
    mobileTabs.addEventListener('change', updateLayout);
    if (window.ResizeObserver) new ResizeObserver(updateLayout).observe(sidebar);
    else window.addEventListener('resize', updateLayout);
    updateLayout();
    $$('.view').forEach((view) => { view.tabIndex = 0; });
    list.addEventListener('click', (ev) => { const b = ev.target.closest('[role="tab"]'); if (b) setTab(b.dataset.tab); });
    $('#tab-audit').addEventListener('click', () => { setTab('audit'); $('#view-audit').focus({ preventScroll: true }); });
    const NAV = TABS.filter((t) => $('#tab-' + t).getAttribute('role') === 'tab');
    list.addEventListener('keydown', (ev) => {
      const i = Math.max(0, NAV.indexOf(S.tab));
      const fwd = NAV[(i + 1) % NAV.length], back = NAV[(i + NAV.length - 1) % NAV.length];
      const direction = mobileTabs.matches ? { ArrowRight: fwd, ArrowLeft: back } : { ArrowDown: fwd, ArrowUp: back };
      const next = { ...direction, Home: NAV[0], End: NAV[NAV.length - 1] }[ev.key];
      if (next) { ev.preventDefault(); setTab(next, true); }
    });
  }

  // ------------------------------------------------------------------ //
  // Dashboard boot
  // ------------------------------------------------------------------ //
  async function initDashboard() {
    $('#app').hidden = false;
    $$('[data-icon]').forEach((el) => el.replaceWith(icon(el.dataset.icon)));
    document.body.append(h('div', { class: 'capture-sink', 'aria-hidden': 'true' }, capVideo));
    initCameraButtons();
    initTabs(); initSettings(); initLive(); initStress(); initAb(); initFairness(); initReview();
    $('#audit-refresh').addEventListener('click', loadAudit);
    $('#audit-limit').addEventListener('change', loadAudit);
    $('#audit-verify').addEventListener('click', verifyAudit);
    $('#audit-json').addEventListener('click', () => exportAudit('json'));
    $('#audit-csv').addEventListener('click', () => exportAudit('csv'));
    on('camera', onCameraChanged);
    // The activity log marks banned people, so redraw it when the people list arrives or changes.
    on('subjects', () => { if (S.tab === 'audit') loadAudit(); });
    window.addEventListener('beforeunload', () => { if (room.info) roomSignal({ type: 'bye', reason: 'ended' }); });
    $$('[data-goto]').forEach((b) => b.addEventListener('click', () => {
      setTab(b.dataset.goto); $('#view-' + b.dataset.goto).focus({ preventScroll: true });
    }));
    $$('.home__section, .fv-sec').forEach((section) => onEnter(section, (delay) => {
      section.style.animationDelay = delay + 'ms';
      section.classList.add('is-entering');
    }));
    on('subjects', () => { if (S.tab === 'home') loadHome(); });
    renderEngine();
    setTab((location.hash || '#home').slice(1));
    pollHealth();
    if (await loadConfig()) loadSubjects();
  }

  // ------------------------------------------------------------------ //
  // Participant (guest) screen: /?room=<id>
  // ------------------------------------------------------------------ //
  function initGuest(roomId) {
    document.title = 'Join call — Fairview';
    $('.skip-link').href = '#guest';
    $('#guest').hidden = false;
    const g = { stream: null, ws: null, pc: null, ice: [], want: false, attempt: 0, timer: null, queue: [], fullRetries: 0, fatal: false };
    const status = $('#guest-status'), conn = $('#guest-conn'), joinBtn = $('#guest-join');
    const say = (text, kind) => setStatus(status, text, kind);
    const signal = (msg) => { if (g.ws && g.ws.readyState === 1) g.ws.send(JSON.stringify(msg)); };
    function dropPeer() { closePeer(g.pc); g.pc = null; g.queue = []; $('#guest-remote').srcObject = null; }
    function fatal(text) {
      g.fatal = true; g.want = false; clearTimeout(g.timer);
      dropPeer();
      if (g.ws) { const ws = g.ws; g.ws = null; ws.close(); }
      setConn(conn, 'failed', 'Not connected'); say(text, 'error');
    }
    function newPeer() {
      const pc = new RTCPeerConnection({ iceServers: g.ice });
      g.pc = pc;
      if (g.stream) g.stream.getVideoTracks().forEach((t) => pc.addTrack(t, g.stream));
      pc.onicecandidate = (ev) => { if (ev.candidate) signal({ type: 'candidate', candidate: ev.candidate.toJSON() }); };
      pc.ontrack = (ev) => { if (g.pc !== pc) return; const v = $('#guest-remote'); v.srcObject = ev.streams[0] || new MediaStream([ev.track]); v.play().catch(() => {}); };
      const update = () => {
        if (g.pc !== pc) return;
        const state = peerState(pc);
        setConn(conn, state, state === 'failed' ? 'Connection failed — use Reconnect' : undefined);
        if (state === 'connected') say('You are connected. The host is analysing your video live.', 'ok');
      };
      pc.onconnectionstatechange = update; pc.oniceconnectionstatechange = update;
      pc.refreshState = update;
      return pc;
    }
    async function onOffer(msg) {
      const desc = { type: 'offer', sdp: sdpOf(msg) };
      let pc = g.pc;
      if (pc && pc.signalingState !== 'closed') {
        try { await pc.setRemoteDescription(desc); } catch (e) { dropPeer(); pc = null; }
      } else pc = null;
      if (!pc) { pc = newPeer(); await pc.setRemoteDescription(desc); }
      for (const c of g.queue.splice(0)) await pc.addIceCandidate(c).catch(() => {});
      await pc.setLocalDescription(await pc.createAnswer());
      signal({ type: 'answer', sdp: pc.localDescription.sdp });
      pc.refreshState(); // an ICE restart on a healthy link fires no state change
    }
    async function onMessage(msg) {
      if (msg.type === 'joined') {
        g.attempt = 0; g.fullRetries = 0;
        setConn(conn, 'waiting', msg.peer_present ? 'Connecting…' : 'Waiting for the host');
        say(msg.peer_present ? 'In the room. Setting up the video link with the host…' : 'In the room. Waiting for the host to connect…');
      } else if (msg.type === 'peer-joined') say('The host is here. Setting up the video link…');
      else if (msg.type === 'offer') await onOffer(msg);
      else if (msg.type === 'candidate' && msg.candidate) {
        if (g.pc && g.pc.remoteDescription) await g.pc.addIceCandidate(msg.candidate).catch(() => {}); else g.queue.push(msg.candidate);
      } else if (msg.type === 'peer-left') {
        dropPeer(); setConn(conn, 'waiting', 'Host disconnected'); say('The host left. Waiting for the host to return…');
      } else if (msg.type === 'bye') {
        dropPeer();
        if (msg.reason === 'renegotiate') { setConn(conn, 'connecting'); say('Setting up the video link…'); }
        else { setConn(conn, 'waiting', 'Call ended by the host'); say('The host ended the call. You can leave, or wait in case the host reconnects.'); }
      } else if (msg.type === 'error') {
        if (msg.code === 'room-not-found') fatal('This room does not exist or has expired. Ask the host for a new link.');
        else if (msg.code === 'room-closed') { dropPeer(); fatal('The host ended the call and closed this room.'); }
        else if (msg.code === 'room-full') {
          if (g.fullRetries++ < 3) { g.retryFull = true; } else fatal('Another participant is already connected to this room.');
        }
      }
    }
    function connect() {
      if (g.ws || !g.want) return;
      clearTimeout(g.timer);
      const ws = new WebSocket(wsUrl(`/ws/room/${encodeURIComponent(roomId)}?role=guest`));
      g.ws = ws;
      ws.addEventListener('message', (ev) => {
        let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
        onMessage(msg).catch((err) => say(`Call setup problem: ${err.message}`, 'error'));
      });
      ws.addEventListener('close', () => {
        if (g.ws !== ws) return;
        g.ws = null;
        if (!g.want || g.fatal) return;
        const delay = g.retryFull ? 800 : backoff(g.attempt++);
        g.retryFull = false;
        if (!g.pc || peerState(g.pc) !== 'connected') { setConn(conn, 'connecting', 'Reconnecting…'); say(`Link to the server lost. Reconnecting in ${Math.max(1, Math.round(delay / 1000))} s…`, 'error'); }
        g.timer = setTimeout(connect, delay);
      });
    }
    async function join() {
      joinBtn.disabled = true;
      say('Requesting camera access…');
      try {
        if (!g.stream) g.stream = await openCamera();
      } catch (err) { say(err.message, 'error'); joinBtn.disabled = false; return; }
      const local = $('#guest-local'); local.srcObject = g.stream; local.play().catch(() => {});
      try { g.ice = (await api('/api/rtc-config?room=' + encodeURIComponent(roomId))).ice_servers || []; }
      catch (err) { g.ice = []; }
      $('#guest-intro').hidden = true; $('#guest-call').hidden = false;
      $('#guest-call').focus();
      g.want = true; g.fatal = false; g.attempt = 0; g.fullRetries = 0;
      say('Joining the room…');
      connect();
      joinBtn.disabled = false;
    }
    function leave() {
      signal({ type: 'bye', reason: 'ended' });
      g.want = false; clearTimeout(g.timer);
      dropPeer();
      if (g.ws) { const ws = g.ws; g.ws = null; ws.close(1000); }
      if (g.stream) { g.stream.getTracks().forEach((t) => t.stop()); g.stream = null; }
      $('#guest-local').srcObject = null;
      $('#guest-call').hidden = true; $('#guest-intro').hidden = false;
      joinBtn.focus();
      setConn(conn, 'idle');
      say('You left the call. Your camera is off. You can close this page or join again.');
      joinBtn.focus();
    }
    joinBtn.addEventListener('click', join);
    $('#guest-leave').addEventListener('click', leave);
    $('#guest-reconnect').addEventListener('click', async () => {
      if (!g.stream) { join(); return; }
      say('Reconnecting…'); setConn(conn, 'connecting', 'Reconnecting…');
      g.want = false; clearTimeout(g.timer);
      dropPeer();
      if (g.ws) { const ws = g.ws; g.ws = null; ws.close(1000); }
      await wait(500); // let the server release this participant slot before rejoining
      g.want = true; g.fatal = false; g.attempt = 0; g.fullRetries = 0;
      connect();
    });
    window.addEventListener('pagehide', () => signal({ type: 'bye', reason: 'ended' }));
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      say('Camera access is unavailable in this context. Browsers only expose the camera on HTTPS or on localhost — ask the host for an https:// link.', 'error');
    }
    apiRaw('/api/rooms/' + encodeURIComponent(roomId)).then((r) => {
      if (r.status === 404) { say('This room does not exist or has expired. Ask the host for a new link.', 'error'); joinBtn.disabled = true; }
    }).catch(() => say('The server could not be reached. Check your network connection and reload the page.', 'error'));
  }

  // ------------------------------------------------------------------ //
  const roomParam = new URLSearchParams(location.search).get('room');
  if (roomParam) initGuest(roomParam); else initDashboard();
})();
