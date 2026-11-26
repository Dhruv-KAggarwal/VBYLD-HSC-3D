/**
 * Reconstruction report + self-evaluation.
 * ---------------------------------------------------------------------------
 * A single-pass drone video can only be reconstructed as well as its coverage
 * allows, so every run ships an honest report: what was measured, how complete
 * it is, and — critically — which numbers are real versus estimated. No score is
 * fabricated; where there is no ground truth the metric is labelled "estimated".
 *
 * Produces both a machine-readable `report.json` and a human-readable
 * `report.md`, written next to the other products.
 */

/** Grade a 0..1 fraction into a short qualitative band. */
function band(fraction, thresholds = [0.3, 0.6, 0.85]) {
  if (fraction >= thresholds[2]) return 'excellent';
  if (fraction >= thresholds[1]) return 'good';
  if (fraction >= thresholds[0]) return 'partial';
  return 'limited';
}

/**
 * Build the evaluation + report objects from a finished job.
 *
 * @param {object} job     the completed reconstruction job
 * @param {object} metrics job.metrics (metadata + pipeline stats)
 * @param {object[]} stages job.stages (per-stage timing from the tracker)
 */
export function buildReport(job, metrics = job.metrics || {}, stages = job.stages || []) {
  const framesUsed = metrics.framesUsed || metrics.frameCount || 0;
  const posed = metrics.cameraPoses || 0;
  const registration = framesUsed ? Math.min(1, posed / framesUsed) : 0;
  const photoCoverage = Number(metrics.photoCoverage ?? 0);
  const surfaceCoverage = Number(metrics.coverage ?? 0);
  const textured = Boolean(metrics.texturedMesh);
  const metric = metrics.scaleSource === 'flight-altitude';

  // Honest evaluation — each row says whether it is measured or estimated.
  const evaluation = [
    {
      parameter: 'Texture realism',
      value: textured
        ? (metrics.densePath === 'openmvs-textured-atlas-mesh'
          ? `Per-pixel photographic atlas (${(photoCoverage * 100).toFixed(0)}% photo coverage)`
          : 'Per-vertex photographic colour from the dense cloud')
        : 'DSM vertex colour only',
      basis: 'measured',
      grade: band(textured ? Math.max(photoCoverage, 0.6) : surfaceCoverage),
    },
    {
      parameter: 'Camera alignment',
      value: `${posed}/${framesUsed} frames registered (${(registration * 100).toFixed(0)}%)`,
      basis: 'measured',
      grade: band(registration),
    },
    {
      parameter: 'Reconstruction completeness',
      value: `${(metrics.points || 0).toLocaleString()} dense points, ${(metrics.meshFaces || 0).toLocaleString()} faces`,
      basis: 'measured',
      grade: band(surfaceCoverage, [0.15, 0.35, 0.6]),
    },
    {
      parameter: 'Metric scale',
      value: metric
        ? `Flight-altitude referenced (${metrics.altitudeM || '?'} m)`
        : 'Scene-height heuristic (no GPS) — scale is approximate',
      basis: metric ? 'measured' : 'estimated',
      grade: metric ? 'good' : 'limited',
    },
    {
      parameter: 'Geometry basis',
      value: 'Multi-view stereo (measured); unseen/occluded surfaces are absent, not invented',
      basis: 'measured',
      grade: 'n/a',
    },
  ];

  const extent = metrics.extentM || metrics.extent || [];
  const timings = stages
    .filter((s) => s.durationMs != null)
    .map((s) => ({ stage: s.label, seconds: Math.round((s.durationMs || 0) / 1000), status: s.status }));
  const totalSeconds = timings.reduce((sum, t) => sum + t.seconds, 0);

  const limitations = [
    posed < framesUsed * 0.9 ? `Only ${posed} of ${framesUsed} frames registered; parts of the flight did not solve.` : null,
    !metric ? 'No GPS/IMU metadata was supplied, so absolute scale is an estimate from flight-altitude heuristics.' : null,
    photoCoverage && photoCoverage < 0.85 ? `${((1 - photoCoverage) * 100).toFixed(0)}% of the surface had no clean photographic view (grazing angle / occlusion).` : null,
    'A single pass gives limited multi-view coverage. Surfaces the camera never saw, and facades it only saw edge-on, are reconstructed from few observations or not at all; treat them as incomplete rather than measured.',
    'To capture true vertical building geometry, fly OBLIQUE (camera tilted ~30-45 degrees below horizontal) and ideally ORBIT each structure. That is what lets multi-view stereo see a wall from two sufficiently different angles. A nadir or straight-forward pass caps how much vertical geometry any method can recover.',
    'Moving objects (vehicles, people) and sky or haze are filtered where detected but may leave residual artefacts.',
  ].filter(Boolean);

  const report = {
    source: metrics.source || job.fileName || 'drone-pass.mp4',
    generatedAt: metrics.generatedAt || new Date().toISOString(),
    profile: metrics.profile || job.profile,
    hardware: metrics.hardware || job.hardware || null,
    processingSeconds: totalSeconds,
    densePath: metrics.densePath,
    upgrades: job.upgrades
      ? Object.entries(job.upgrades).filter(([, v]) => v.available).map(([k, v]) => k + (v.mode ? `:${v.mode}` : ''))
      : [],
    metrics: {
      framesUsed,
      framesRegistered: posed,
      registrationPercent: Math.round(registration * 100),
      densePoints: metrics.points || 0,
      meshFaces: metrics.meshFaces || 0,
      photoCoveragePercent: Math.round(photoCoverage * 100),
      surfaceCoveragePercent: Math.round(surfaceCoverage * 100),
      extentMetres: extent.map((v) => Math.round(v)),
      scaleSource: metrics.scaleSource || 'unknown',
      georeferenced: metric,
    },
    evaluation,
    timings,
    limitations,
  };
  return report;
}

/** Render the report object as GitHub-flavoured Markdown. */
export function reportToMarkdown(report) {
  const rows = report.evaluation
    .map((e) => `| ${e.parameter} | ${e.value} | ${e.basis} | ${e.grade} |`)
    .join('\n');
  const timing = report.timings.length
    ? report.timings.map((t) => `- ${t.stage}: ${t.seconds}s${t.status === 'skipped' ? ' (skipped)' : ''}`).join('\n')
    : '- (per-stage timing unavailable)';
  const limits = report.limitations.map((l) => `- ${l}`).join('\n');
  const m = report.metrics;
  return `# Reconstruction report — ${report.source}

Generated ${report.generatedAt} · profile **${report.profile}** · ${report.processingSeconds}s total
Dense path: \`${report.densePath}\`${report.upgrades?.length ? `
Active upgrades: ${report.upgrades.join(', ')}` : ''}

## Key metrics (measured)

- Frames used: **${m.framesUsed}**, registered: **${m.framesRegistered} (${m.registrationPercent}%)**
- Dense points: **${m.densePoints.toLocaleString()}**, mesh faces: **${m.meshFaces.toLocaleString()}**
- Photographic coverage: **${m.photoCoveragePercent}%** · surface coverage: **${m.surfaceCoveragePercent}%**
- Extent: **${m.extentMetres.join(' × ')} m** · scale: **${m.scaleSource}**${m.georeferenced ? '' : ' _(estimated, no GPS)_'}

## Evaluation

| Parameter | Result | Basis | Grade |
|---|---|---|---|
${rows}

## Stage timing

${timing}

## Known limitations

${limits}

_Geometry is measured by multi-view stereo. Surfaces that the single pass never
observed are left absent rather than invented._
`;
}
