import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import {
  LayoutDashboard,
  Users,
  Calendar,
  FileText,
  Settings,
  Search,
  Scan,
  Clock,
  TrendingUp,
  ChevronRight,
  Activity,
  Bell,
} from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

/* ── Mock data ─────────────────────────────────────────────────────────── */
const statCards = [
  { icon: Scan,       label: 'Total Scans',     value: '1,247', change: '+12%', changeUp: true,  accent: 'mint'  },
  { icon: Clock,      label: 'Pending Review',   value: '13',    change: '−3',   changeUp: false, accent: 'amber' },
  { icon: TrendingUp, label: 'Accuracy',          value: '98.6%', change: '+0.4%',changeUp: true,  accent: 'emerald'},
  { icon: Activity,   label: 'Scans Today',       value: '34',    change: '+8',   changeUp: true,  accent: 'sky'   },
];

const recentActivity = [
  { initials: 'SP', name: 'S. Patel',    diagnosis: 'Mild DR detected',       time: '4 min ago',  status: 'warning' },
  { initials: 'AK', name: 'A. Kumar',    diagnosis: 'Normal — no pathology',  time: '18 min ago', status: 'healthy' },
  { initials: 'RJ', name: 'R. Joshi',    diagnosis: 'Glaucoma suspected',     time: '42 min ago', status: 'critical'},
  { initials: 'MG', name: 'M. Gupta',    diagnosis: 'Myopia flagged',         time: '1 hr ago',   status: 'warning' },
];

const weeklyData = [22, 35, 28, 41, 37, 52, 34]; // Mon–Sun
const dayLabels  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

/* ── Helpers ───────────────────────────────────────────────────────────── */
const statusDot: Record<string, string> = {
  healthy:  'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.7)]',
  warning:  'bg-amber-400   shadow-[0_0_6px_rgba(251,191,36,0.7)]',
  critical: 'bg-rose-400    shadow-[0_0_6px_rgba(251,113,133,0.7)]',
};

const accentColor: Record<string, string> = {
  mint:    'rgba(39,209,127,0.15)',
  amber:   'rgba(251,191,36,0.12)',
  emerald: 'rgba(52,211,153,0.12)',
  sky:     'rgba(56,189,248,0.12)',
};
const accentText: Record<string, string> = {
  mint:    'text-[#27D17F]',
  amber:   'text-amber-400',
  emerald: 'text-emerald-400',
  sky:     'text-sky-400',
};

function buildAreaPath(data: number[], w: number, h: number) {
  const maxVal = Math.max(...data) * 1.15;
  const step   = w / (data.length - 1);
  const pts    = data.map((v, i) => [i * step, h - (v / maxVal) * h] as const);

  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length; i++) {
    const cpx1 = pts[i - 1][0] + step * 0.4;
    const cpx2 = pts[i][0]     - step * 0.4;
    d += ` C${cpx1},${pts[i - 1][1]} ${cpx2},${pts[i][1]} ${pts[i][0]},${pts[i][1]}`;
  }

  const area = d + ` L${pts[pts.length - 1][0]},${h} L${pts[0][0]},${h} Z`;
  return { line: d, area };
}

/* ── Component ─────────────────────────────────────────────────────────── */
export default function DashboardSection() {
  const sectionRef   = useRef<HTMLElement>(null);
  const dashboardRef = useRef<HTMLDivElement>(null);
  const sidebarRef   = useRef<HTMLDivElement>(null);
  const statsRef     = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<HTMLDivElement>(null);
  const listRef      = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section   = sectionRef.current;
    const dashboard = dashboardRef.current;
    const sidebar   = sidebarRef.current;
    const stats     = statsRef.current;
    const chart     = chartRef.current;
    const list      = listRef.current;

    if (!section || !dashboard || !sidebar || !stats || !chart || !list) return;

    const ctx = gsap.context(() => {
      gsap.set(dashboard,        { opacity: 0, y: -60, scale: 0.96 });
      gsap.set(sidebar,          { opacity: 0, x: -30 });
      gsap.set(stats.children,   { opacity: 0, y: -30 });
      gsap.set(chart,            { opacity: 0, x: -40 });
      gsap.set(list,             { opacity: 0, x: 40 });

      // Fade out as next section scrolls over
      gsap.to(section, {
        opacity: 0,
        y: -40,
        scrollTrigger: {
          trigger: section,
          start: 'bottom bottom',
          end: 'bottom 20%',
          scrub: 0.3,
        },
      });

      ScrollTrigger.create({
        trigger: section,
        start: 'top 70%',
        onEnter: () => {
          const tl = gsap.timeline({ defaults: { ease: 'power2.out', duration: 0.7 } });
          tl.to(dashboard, { y: 0, scale: 1, opacity: 1 }, 0)
            .to(sidebar,   { x: 0, opacity: 1 }, 0.08)
            .to(stats.children, { y: 0, opacity: 1, stagger: 0.06 }, 0.12)
            .to(chart,     { x: 0, opacity: 1 }, 0.2)
            .to(list,      { x: 0, opacity: 1 }, 0.2);
        },
        once: true,
      });
    }, section);

    return () => ctx.revert();
  }, []);

  /* sidebar items */
  const sidebarItems = [
    { icon: LayoutDashboard, label: 'Home',     active: true  },
    { icon: Users,           label: 'Patients',  active: false },
    { icon: Calendar,        label: 'Schedule',  active: false },
    { icon: FileText,        label: 'Reports',   active: false },
    { icon: Settings,        label: 'Settings',  active: false },
  ];

  /* chart SVG */
  const chartW = 380;
  const chartH = 120;
  const { line, area } = buildAreaPath(weeklyData, chartW, chartH);

  return (
    <section
      ref={sectionRef}
      id="dashboard"
      className="relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ position: 'sticky', top: 0, zIndex: 6, backgroundColor: 'rgba(8,15,23,0.55)' }}
    >
      {/* Dashboard Card */}
      <div
        ref={dashboardRef}
        className="absolute rounded-[28px] card-shadow border border-white/[0.06] overflow-hidden"
        style={{
          left: '10vw', top: '14vh',
          width: '80vw', height: '72vh',
          background: 'linear-gradient(145deg, #0d1a28 0%, #0a1520 100%)',
        }}
      >
        {/* subtle dot-grid */}
        <div className="absolute inset-0 dot-pattern opacity-[0.06] pointer-events-none" />
        {/* Gradient vignette */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ background: 'radial-gradient(ellipse at 30% 20%, rgba(39,209,127,0.04) 0%, transparent 60%)' }}
        />

        {/* ── Sidebar ──────────────────────────────────────────────────────── */}
        <div
          ref={sidebarRef}
          className="absolute left-0 top-0 bottom-0 w-[68px] backdrop-blur-sm border-r border-white/[0.06] flex flex-col items-center py-6 gap-1.5"
          style={{ background: 'rgba(255,255,255,0.03)' }}
        >
          {/* Brand icon */}
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-5" style={{ background: 'rgba(39,209,127,0.12)', boxShadow: '0 0 18px rgba(39,209,127,0.15)' }}>
            <Scan className="w-5 h-5 text-[#27D17F]" />
          </div>
          {sidebarItems.map((item) => (
            <button
              key={item.label}
              title={item.label}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                item.active
                  ? 'bg-[rgba(39,209,127,0.12)] text-[#27D17F]'
                  : 'text-white/25 hover:text-white/50 hover:bg-white/[0.04]'
              }`}
              style={item.active ? { boxShadow: '0 0 12px rgba(39,209,127,0.18)' } : undefined}
            >
              <item.icon className="w-[18px] h-[18px]" />
            </button>
          ))}
        </div>

        {/* ── Main Content ─────────────────────────────────────────────────── */}
        <div className="absolute left-[68px] right-0 top-0 bottom-0 p-6 overflow-auto">

          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-white/90 mb-0.5" style={{ fontFamily: "'Sora', sans-serif" }}>
                Dashboard
              </h2>
              <p className="text-xs text-white/35">Welcome back, Dr. Reddy</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-white/30" />
                <input
                  type="text"
                  placeholder="Search patients…"
                  className="pl-9 pr-4 py-2 rounded-xl text-xs text-white placeholder:text-white/30 border border-white/[0.06] focus:outline-none focus:border-[#27D17F]/40 w-52 transition-colors"
                  style={{ background: 'rgba(255,255,255,0.04)' }}
                />
              </div>
              <button
                className="w-9 h-9 rounded-xl border border-white/[0.06] flex items-center justify-center text-white/40 hover:text-white/60 hover:border-white/10 transition-colors relative"
                style={{ background: 'rgba(255,255,255,0.04)' }}
              >
                <Bell className="w-4 h-4" />
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-rose-400" style={{ boxShadow: '0 0 6px rgba(251,113,133,0.6)' }} />
              </button>
              <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'rgba(39,209,127,0.12)', boxShadow: '0 0 14px rgba(39,209,127,0.12)' }}>
                <span className="text-xs font-bold text-[#27D17F]">DR</span>
              </div>
            </div>
          </div>

          {/* ── Stats Row ───────────────────────────────────────────────────── */}
          <div ref={statsRef} className="grid grid-cols-4 gap-3 mb-5">
            {statCards.map((stat, i) => (
              <div
                key={i}
                className="rounded-2xl p-4 border border-white/[0.06] backdrop-blur-sm transition-all duration-300 hover:border-white/[0.10] group"
                style={{ background: 'rgba(255,255,255,0.03)' }}
              >
                <div className="flex items-center justify-between mb-3">
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center transition-transform duration-300 group-hover:scale-110"
                    style={{ background: accentColor[stat.accent] }}
                  >
                    <stat.icon className={`w-4 h-4 ${accentText[stat.accent]}`} />
                  </div>
                  <span className={`text-[10px] font-semibold ${stat.changeUp ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {stat.change}
                  </span>
                </div>
                <div className="text-xl font-bold text-white leading-none mb-1">{stat.value}</div>
                <span className="text-[11px] text-white/35">{stat.label}</span>
              </div>
            ))}
          </div>

          {/* ── Bottom Row ──────────────────────────────────────────────────── */}
          <div className="grid grid-cols-5 gap-3" style={{ height: 'calc(100% - 180px)', minHeight: '200px' }}>

            {/* Chart Card — spans 3 */}
            <div
              ref={chartRef}
              className="col-span-3 rounded-2xl p-5 border border-white/[0.06] backdrop-blur-sm flex flex-col"
              style={{ background: 'rgba(255,255,255,0.03)' }}
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white/80">Scans Overview</h3>
                  <p className="text-[10px] text-white/30 mt-0.5">This week vs. last week</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#27D17F]" />
                  <span className="text-[10px] text-white/40">This week</span>
                </div>
              </div>

              {/* SVG Chart */}
              <div className="flex-1 relative min-h-0">
                <svg
                  viewBox={`0 0 ${chartW} ${chartH}`}
                  preserveAspectRatio="none"
                  className="w-full h-full"
                >
                  <defs>
                    <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor="rgba(39,209,127,0.30)" />
                      <stop offset="100%" stopColor="rgba(39,209,127,0.00)" />
                    </linearGradient>
                  </defs>
                  {/* Grid lines */}
                  {[0.25, 0.5, 0.75].map((frac) => (
                    <line
                      key={frac}
                      x1="0" y1={chartH * frac} x2={chartW} y2={chartH * frac}
                      stroke="rgba(255,255,255,0.04)" strokeWidth="1"
                    />
                  ))}
                  {/* Filled area */}
                  <path d={area} fill="url(#chartGrad)" />
                  {/* Line */}
                  <path d={line} fill="none" stroke="#27D17F" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="chart-line-draw" />
                  {/* Data dots */}
                  {weeklyData.map((v, i) => {
                    const maxVal = Math.max(...weeklyData) * 1.15;
                    const cx = (i / (weeklyData.length - 1)) * chartW;
                    const cy = chartH - (v / maxVal) * chartH;
                    return (
                      <circle key={i} cx={cx} cy={cy} r="3.5" fill="#27D17F" stroke="#0d1a28" strokeWidth="2" />
                    );
                  })}
                </svg>
                {/* Day labels */}
                <div className="flex justify-between mt-1.5 px-0.5">
                  {dayLabels.map((d) => (
                    <span key={d} className="text-[9px] text-white/25">{d}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Activity Card — spans 2 */}
            <div
              ref={listRef}
              className="col-span-2 rounded-2xl p-5 border border-white/[0.06] backdrop-blur-sm flex flex-col"
              style={{ background: 'rgba(255,255,255,0.03)' }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-white/80">Recent Activity</h3>
                <button className="text-[10px] text-[#27D17F] flex items-center gap-0.5 hover:underline">
                  View all <ChevronRight className="w-3 h-3" />
                </button>
              </div>
              <div className="flex-1 space-y-2.5 overflow-auto">
                {recentActivity.map((item, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-3 p-2.5 rounded-xl transition-colors hover:bg-white/[0.03] group/row"
                  >
                    {/* Avatar */}
                    <div
                      className="w-9 h-9 rounded-xl flex items-center justify-center flex-none"
                      style={{ background: 'rgba(39,209,127,0.10)' }}
                    >
                      <span className="text-[10px] font-bold text-[#27D17F]">{item.initials}</span>
                    </div>
                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-white/75 truncate">{item.name}</p>
                      <p className="text-[10px] text-white/35 truncate">{item.diagnosis}</p>
                    </div>
                    {/* Meta */}
                    <div className="flex items-center gap-2 flex-none">
                      <span className="text-[10px] text-white/25">{item.time}</span>
                      <span className={`w-2 h-2 rounded-full ${statusDot[item.status]}`} />
                    </div>
                  </div>
                ))}

                {/* Quick-stat mini bar */}
                <div className="pt-2 mt-auto border-t border-white/[0.04]">
                  <div className="flex items-center justify-between text-[10px] text-white/30 mb-2">
                    <span>Disease distribution today</span>
                    <span className="text-white/50">34 scans</span>
                  </div>
                  <div className="flex gap-1 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-amber-400/70 rounded-full" style={{ width: '35%' }} title="DR" />
                    <div className="bg-rose-400/70 rounded-full"  style={{ width: '15%' }} title="Glaucoma" />
                    <div className="bg-sky-400/70 rounded-full"   style={{ width: '20%' }} title="Myopia" />
                    <div className="bg-emerald-400/70 rounded-full" style={{ width: '30%' }} title="Normal" />
                  </div>
                  <div className="flex gap-3 mt-1.5">
                    {[
                      { color: 'bg-amber-400',   label: 'DR' },
                      { color: 'bg-rose-400',    label: 'Glaucoma' },
                      { color: 'bg-sky-400',     label: 'Myopia' },
                      { color: 'bg-emerald-400', label: 'Normal' },
                    ].map((l) => (
                      <div key={l.label} className="flex items-center gap-1">
                        <span className={`w-1.5 h-1.5 rounded-full ${l.color}`} />
                        <span className="text-[9px] text-white/30">{l.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
