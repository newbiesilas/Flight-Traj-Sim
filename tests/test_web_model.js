/* Validate the JavaScript model embedded in web/index.html by executing the
 * exact same source in Node.  Run:  node tests/test_web_model.js
 *
 * The checks fall into three groups:
 *   1. tabulated ISA values,
 *   2. physical behaviour of the full mission (range, endurance, fuel),
 *   3. monotonicity / sensitivity - the model must move the right way when a
 *      parameter is perturbed.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8');
const begin = html.indexOf('/* ===== MODEL:BEGIN =====');
const end = html.indexOf('/* ===== MODEL:END ===== */');
if (begin < 0 || end < 0) {
  console.error('FAIL: could not locate the MODEL markers in web/index.html');
  process.exit(1);
}
const source = html.slice(begin, end);

const sandbox = { console, Math, isFinite, isNaN, Infinity, Object, Array };
vm.createContext(sandbox);
vm.runInContext(source, sandbox);

let failures = 0;

function check(name, actual, expected, tol) {
  const ok = Number.isFinite(actual) && Math.abs(actual - expected) <= tol;
  if (!ok) failures++;
  console.log(
    (ok ? 'PASS' : 'FAIL') + '  ' + name +
    '  actual=' + Number(actual).toFixed(4) + ' expected=' + expected.toFixed(4) + ' tol=' + tol
  );
}

function assert(name, cond, detail) {
  if (!cond) failures++;
  console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (detail ? '  (' + detail + ')' : ''));
}

// ---------------------------------------------------------------- ISA ----
/* The ISA tables are tabulated in geopotential altitude; geometric() converts
   a geopotential height back to the geometric altitude the functions take. */
const geometric = (H) => 6356766 * H / (6356766 - H);
console.log('--- International Standard Atmosphere ---');
check('density @ H=0', sandbox.density(0), 1.22500, 0.005);
check('density @ H=5000 m', sandbox.density(geometric(5000)), 0.73612, 0.006);
check('density @ H=11000 m', sandbox.density(geometric(11000)), 0.36392, 0.004);
check('density @ H=15000 m', sandbox.density(geometric(15000)), 0.19367, 0.003);
check('density @ H=20000 m', sandbox.density(geometric(20000)), 0.08803, 0.002);
check('pressure @ H=11000 m', sandbox.pressure(geometric(11000)), 22632.0, 20);
check('pressure @ H=20000 m', sandbox.pressure(geometric(20000)), 5474.9, 10);
check('speed of sound @ H=0', sandbox.speedOfSound(0), 340.29, 0.5);
check('speed of sound @ H=11000 m', sandbox.speedOfSound(geometric(11000)), 295.07, 0.5);

// ------------------------------------------------------------ mission ----
const p = {
  m0: 12000, fuel: 3000, S: 24, cd0: 0.025, k: 0.048,
  thrust: 39000, tsfc: 1.6e-5, hcruise: 11000, cruiseMach: 0.78,
  climbCas: 150, descentCas: 150, descentThrottle: 0.15,
  clMax: 1.4, mcrit: 0.82, lapse: 0.9, reserve: 200,
  climbThrottle: 1.0, maxClimbGammaDeg: 8.0, climbMinGammaDeg: -10.0,
  minDescentGammaDeg: -20.0, stallPatience: 600,
  h0: 100, hEnd: 500, dt: 2.0, tMax: 40000, nPoints: 400
};

console.log('\n--- Baseline mission ---');
const r = sandbox.simulate(p);
console.log('range=' + r.range_km.toFixed(0) + ' km  endurance=' + r.endurance_min.toFixed(0) +
  ' min  fuel=' + r.fuel_kg.toFixed(0) + ' kg  completed=' + r.completed);

assert('mission completes', r.completed === true, r.note);
assert('three phases recorded', r.phases.length === 3,
  r.phases.map((x) => x.name).join('/'));
assert('range is physically plausible (500-12000 km)',
  r.range_km > 500 && r.range_km < 12000, r.range_km.toFixed(0) + ' km');
assert('endurance is plausible (30-900 min)',
  r.endurance_min > 30 && r.endurance_min < 900, r.endurance_min.toFixed(0) + ' min');
assert('fuel burned close to the usable fuel',
  Math.abs(r.fuel_kg - (p.fuel - p.reserve)) < 150, r.fuel_kg.toFixed(0) + ' kg');
assert('descends to the end altitude', Math.abs(r.h[r.h.length - 1] - p.hEnd) < 50,
  r.h[r.h.length - 1].toFixed(1) + ' m');
assert('altitude never negative', Math.min.apply(null, r.h) >= -1e-6);
assert('climb levels off at the cruise altitude',
  Math.abs(r.h[r.phases[0].i1 - 1] - p.hcruise) < 5);
assert('cruise throttle is inside [0, 1]',
  r.cruiseThrottle > 0 && r.cruiseThrottle < 1, r.cruiseThrottle.toFixed(3));
assert('never stalls', r.max_cl < p.clMax, 'peak CL = ' + r.max_cl.toFixed(3));

// The simulated cruise must reproduce the closed-form Breguet range.
const cr = r.phases.find((x) => x.name === 'cruise');
const cruiseKm = (r.x[cr.i1 - 1] - r.x[cr.i0]) / 1000;
const breguetKm = sandbox.breguetRange(p, r.m[cr.i0], r.m[cr.i1 - 1]) / 1000;
console.log('cruise segment=' + cruiseKm.toFixed(1) + ' km ; Breguet=' + breguetKm.toFixed(1) + ' km');
assert('cruise segment matches Breguet within 1 %',
  Math.abs(cruiseKm - breguetKm) / breguetKm < 0.01,
  (100 * (cruiseKm - breguetKm) / breguetKm).toFixed(3) + ' %');

// ------------------------------------------------------- sensitivity ----
console.log('\n--- Sensitivity ---');
const heavy = sandbox.simulate(Object.assign({}, p, { m0: 15000 }));
const light = sandbox.simulate(Object.assign({}, p, { m0: 10000 }));
console.log('range @10t=' + light.range_km.toFixed(0) + ' km ; @15t=' + heavy.range_km.toFixed(0) + ' km');
assert('lighter aircraft flies further on the same fuel',
  light.range_km > r.range_km && r.range_km > heavy.range_km);

const draggy = sandbox.simulate(Object.assign({}, p, { cd0: 0.030 }));
console.log('range cd0=0.025 -> ' + r.range_km.toFixed(0) +
  ' km ; cd0=0.030 -> ' + draggy.range_km.toFixed(0) + ' km');
assert('higher CD0 reduces range', draggy.range_km < r.range_km);

const draggier = sandbox.simulate(Object.assign({}, p, { cd0: 0.045 }));
assert('range keeps falling as CD0 keeps rising', draggier.range_km < draggy.range_km,
  draggier.range_km.toFixed(0) + ' km');

const induced = sandbox.simulate(Object.assign({}, p, { k: 0.070 }));
assert('higher induced drag factor reduces range', induced.range_km < r.range_km,
  induced.range_km.toFixed(0) + ' km');

const bigWing = sandbox.simulate(Object.assign({}, p, { S: 44 }));
assert('a much larger wing hurts range at fixed Mach (too far below best L/D)',
  bigWing.range_km < r.range_km, bigWing.range_km.toFixed(0) + ' km');

/* Optimum sizing: at a fixed cruise speed and altitude the range peaks where
   CL = sqrt(cd0/k), i.e. exactly where the polar gives (L/D)max. */
const areas = [];
for (let S = 12; S <= 46; S += 2) areas.push(S);
const areaRows = areas.map((S) => sandbox.simulate(Object.assign({}, p, { S: S })).range_km);
let bestIdx = 0;
for (let i = 1; i < areaRows.length; i++) {
  if (areaRows[i] > areaRows[bestIdx]) bestIdx = i;
}
const bestS = areas[bestIdx];
const clOpt = Math.sqrt(p.cd0 / p.k);
const vCruise = sandbox.scheduledSpeed(p.hcruise, p.climbCas, p.cruiseMach);
const qCruise = 0.5 * sandbox.density(p.hcruise) * vCruise * vCruise;
const sOpt = p.m0 * 9.80665 / (qCruise * clOpt);
console.log('best wing area = ' + bestS + ' m2 ; theory (CL = sqrt(cd0/k)) = ' +
  sOpt.toFixed(1) + ' m2');
assert('range peaks near the wing area giving CL = sqrt(cd0/k)',
  Math.abs(bestS - sOpt) / sOpt < 0.30);

const thirsty = sandbox.simulate(Object.assign({}, p, { tsfc: 2.4e-5 }));
assert('higher TSFC reduces range', thirsty.range_km < r.range_km,
  thirsty.range_km.toFixed(0) + ' km');

const moreFuel = sandbox.simulate(Object.assign({}, p, { fuel: 4500 }));
assert('more fuel gives more range', moreFuel.range_km > r.range_km,
  moreFuel.range_km.toFixed(0) + ' km');

const lower = sandbox.simulate(Object.assign({}, p, { hcruise: 6000 }));
assert('a lower cruise altitude costs range (denser air)',
  lower.range_km < r.range_km, lower.range_km.toFixed(0) + ' km');

const weak = sandbox.simulate(Object.assign({}, p, { thrust: 11000 }));
assert('an under-powered aircraft cannot fly the mission',
  weak.completed === false, weak.note);

// ------------------------------------------------------------- sweep ----
console.log('\n--- Sweep helper ---');
const rows = sandbox.sweep(p, 'm0', [9000, 12000, 15000]);
assert('sweep returns one row per value', rows.length === 3);
assert('sweep range decreases with mass',
  rows[0].range_km > rows[1].range_km && rows[1].range_km > rows[2].range_km,
  rows.map((x) => x.range_km.toFixed(0)).join(' > '));

console.log('\n' + (failures === 0 ? 'ALL CHECKS PASSED' : failures + ' CHECK(S) FAILED'));
process.exit(failures === 0 ? 0 : 1);
