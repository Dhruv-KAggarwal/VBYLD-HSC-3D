/**
 * Skyforge structured logger.
 * ---------------------------------------------------------------------------
 * One small module that every stage of the reconstruction pipeline writes
 * through, so the operator sees a single, consistent, timestamped stream of
 * what the machine is doing — and, crucially, *why it stopped* when something
 * fails.
 *
 * Design goals (why this exists at all):
 *
 *   1. Live terminal feedback. A photogrammetry run is minutes of silence per
 *      stage otherwise. Every engine line of interest, every stage boundary and
 *      every error is stamped with wall-clock time and the elapsed time since
 *      the run started, so a frozen pipeline is obvious at a glance.
 *
 *   2. A durable record. The same stream is appended to a log file on disk
 *      (`runtime/<job>/pipeline.log` plus a rolling `logs/skyforge.log`), so a
 *      crash that scrolls off the terminal can still be diagnosed afterwards.
 *
 *   3. Errors are never swallowed. `error()` prints the full message in red to
 *      stderr and records a stack when one is available. Logging itself is
 *      wrapped in try/catch everywhere so a logging failure can never take the
 *      reconstruction down with it.
 *
 * Nothing here throws. If the log file cannot be opened we degrade to
 * terminal-only; if the terminal is not a TTY we drop the ANSI colour codes.
 */

import { appendFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname } from 'node:path';

/** ANSI colour helpers, disabled automatically when output is piped or NO_COLOR is set. */
const useColour = () => process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code, text) => (useColour() ? `[${code}m${text}[0m` : text);
const colours = {
  dim: (t) => paint('2', t),
  grey: (t) => paint('90', t),
  cyan: (t) => paint('36', t),
  green: (t) => paint('32', t),
  yellow: (t) => paint('33', t),
  red: (t) => paint('1;31', t),
  magenta: (t) => paint('35', t),
  bold: (t) => paint('1', t),
};

/** The five levels the pipeline speaks in, each with a fixed-width tag and colour. */
const LEVELS = {
  info: { tag: 'INFO ', colour: colours.cyan, stream: 'stdout' },
  stage: { tag: 'STAGE', colour: colours.magenta, stream: 'stdout' },
  step: { tag: 'STEP ', colour: colours.grey, stream: 'stdout' },
  ok: { tag: 'OK   ', colour: colours.green, stream: 'stdout' },
  warn: { tag: 'WARN ', colour: colours.yellow, stream: 'stderr' },
  error: { tag: 'ERROR', colour: colours.red, stream: 'stderr' },
};

/** `93482ms` -> `1m33s`, so the elapsed column stays narrow for long runs. */
function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes}m${String(seconds).padStart(2, '0')}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h${String(minutes % 60).padStart(2, '0')}m`;
}

function wallClock(date) {
  // Local HH:MM:SS — an operator reading the terminal thinks in wall-clock time,
  // not UTC. The elapsed "+Ns" column carries the timezone-independent measure.
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/**
 * Creates a logger bound to one reconstruction run.
 *
 * @param {object} options
 * @param {string} [options.label]   short run identifier shown in the file log
 * @param {string[]} [options.files] absolute paths the stream is appended to
 * @param {boolean} [options.quiet]  suppress terminal output (file only)
 * @param {() => number} [options.now] clock injection, for tests
 */
export function createLogger({ label = 'run', files = [], quiet = false, now = () => Date.now() } = {}) {
  const startedAt = now();
  const sinks = [];
  for (const file of files) {
    try {
      const dir = dirname(file);
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      appendFileSync(file, `\n===== Skyforge run "${label}" started ${new Date().toISOString()} =====\n`);
      sinks.push(file);
    } catch {
      // A log file we cannot open is not worth failing the run over.
    }
  }

  /** Append one already-rendered line to every file sink. Never throws. */
  const toFiles = (plainLine) => {
    for (const file of sinks) {
      try { appendFileSync(file, `${plainLine}\n`); } catch { /* disk full / locked — ignore */ }
    }
  };

  /** Render `[HH:MM:SS +1m03s] TAG  message` with and without colour. */
  const render = (level, message) => {
    const spec = LEVELS[level] || LEVELS.info;
    const elapsed = formatElapsed(now() - startedAt);
    const stamp = `[${wallClock(new Date())} +${elapsed.padStart(5, ' ')}]`;
    const plain = `${stamp} ${spec.tag} ${message}`;
    const pretty = `${colours.grey(stamp)} ${spec.colour(spec.tag)} ${message}`;
    return { plain, pretty, stream: spec.stream };
  };

  const emit = (level, message) => {
    const { plain, pretty, stream } = render(level, String(message));
    if (!quiet) {
      try { process[stream].write(`${pretty}\n`); } catch { /* closed pipe */ }
    }
    toFiles(plain);
  };

  const logger = {
    startedAt,
    elapsedMs: () => now() - startedAt,

    /** General progress narration. */
    info: (message) => emit('info', message),

    /** A stage boundary — rendered with emphasis so the run reads as chapters. */
    stage: (index, count, name, detail = '') => {
      const prefix = count ? colours.bold(`[${index}/${count}]`) : '';
      emit('stage', `${prefix} ${colours.bold(name)}${detail ? colours.dim(` — ${detail}`) : ''}`.trim());
    },

    /** Sub-stage detail (e.g. "view 45/90 · pass 2/2"). Low-noise grey. */
    step: (message) => emit('step', message),

    /** A stage finished cleanly, with an optional measured duration. */
    ok: (message, durationMs = null) => {
      emit('ok', durationMs != null ? `${message} ${colours.dim(`(${formatElapsed(durationMs)})`)}` : message);
    },

    /** A recoverable problem: the run continues, usually on a fallback path. */
    warn: (message) => emit('warn', colours.yellow(String(message))),

    /**
     * A failure. Prints the message in red to stderr and records the stack in
     * the file log so a post-mortem has everything the terminal scrolled past.
     */
    error: (message, errorObject = null) => {
      emit('error', colours.red(String(message)));
      if (errorObject?.stack) toFiles(errorObject.stack);
    },

    /** A single key=value measurement line, e.g. metrics at the end of a run. */
    metric: (key, value) => emit('info', `${colours.dim(`${key}:`)} ${colours.bold(String(value))}`),

    /** Raw engine output already prefixed by the caller — file only by default. */
    engine: (line, { echo = false } = {}) => {
      const text = String(line).replace(/\s+$/, '');
      if (!text) return;
      toFiles(`        ${text}`);
      if (echo && !quiet) {
        try { process.stdout.write(`${colours.dim(`        ${text}`)}\n`); } catch { /* closed pipe */ }
      }
    },

    /** Lets other code reuse the same timestamp/elapsed formatting. */
    stamp: (level, message) => render(level, message),
  };

  return logger;
}

/**
 * A process-wide default logger so modules that are not handed one (the CLI
 * banner, the server boot line) still speak the same dialect. Reconstruction
 * runs create their own per-job logger with file sinks attached.
 */
export const log = createLogger({ label: 'skyforge' });
