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
} from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function DashboardSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const dashboardRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const dashboard = dashboardRef.current;
    const sidebar = sidebarRef.current;
    const stats = statsRef.current;
    const chart = chartRef.current;
    const list = listRef.current;

    if (!section || !dashboard || !sidebar || !stats || !chart || !list) return;

    const ctx = gsap.context(() => {
      const scrollTl = gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: 'top top',
          end: '+=100%',
          pin: true,
          scrub: 0.4,
        },
      });

      // ENTRANCE (0-40%)
      scrollTl
        .fromTo(
          dashboard,
          { y: '60vh', scale: 0.96, opacity: 0 },
          { y: 0, scale: 1, opacity: 1, ease: 'none' },
          0
        )
        .fromTo(sidebar, { x: '-8vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0.04)
        .fromTo(
          stats.children,
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, stagger: 0.05, ease: 'none' },
          0.08
        )
        .fromTo(chart, { x: '-15vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0.1)
        .fromTo(list, { x: '15vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0.1);

      // EXIT (60-100%)
      scrollTl
        .fromTo(
          dashboard,
          { y: 0, opacity: 1 },
          { y: '-12vh', opacity: 0.3, ease: 'power2.in' },
          0.6
        )
        .fromTo(sidebar, { x: 0, opacity: 1 }, { x: '-4vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(
          stats.children,
          { y: 0, opacity: 1 },
          { y: -12, opacity: 0.25, ease: 'power2.in' },
          0.6
        )
        .fromTo(chart, { x: 0, opacity: 1 }, { x: '-6vw', opacity: 0.25, ease: 'power2.in' }, 0.6)
        .fromTo(list, { x: 0, opacity: 1 }, { x: '6vw', opacity: 0.25, ease: 'power2.in' }, 0.6);
    }, section);

    return () => ctx.revert();
  }, []);

  const sidebarItems = [
    { icon: LayoutDashboard, label: 'Home', active: true },
    { icon: Users, label: 'Patients', active: false },
    { icon: Calendar, label: 'Schedule', active: false },
    { icon: FileText, label: 'Reports', active: false },
    { icon: Settings, label: 'Settings', active: false },
  ];

  return (
    <section
      ref={sectionRef}
      id="dashboard"
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-[60]"
    >
      {/* Dashboard Card */}
      <div
        ref={dashboardRef}
        className="absolute bg-offwhite rounded-[28px] card-shadow overflow-hidden"
        style={{
          left: '10vw',
          top: '14vh',
          width: '80vw',
          height: '72vh',
        }}
      >
        {/* Background Image */}
        <img
          src="/dashboard_room.jpg"
          alt="Dashboard background"
          className="absolute inset-0 w-full h-full object-cover opacity-30"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-offwhite/95 to-offwhite/80" />

        {/* Sidebar */}
        <div
          ref={sidebarRef}
          className="absolute left-0 top-0 bottom-0 w-16 bg-white/80 backdrop-blur-sm border-r border-navy/5 flex flex-col items-center py-6 gap-2"
        >
          <div className="w-10 h-10 rounded-xl bg-mint/15 flex items-center justify-center mb-4">
            <Scan className="w-5 h-5 text-mint" />
          </div>
          {sidebarItems.map((item) => (
            <button
              key={item.label}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
                item.active ? 'bg-mint/15 text-mint' : 'text-navy/40 hover:text-navy hover:bg-navy/5'
              }`}
            >
              <item.icon className="w-5 h-5" />
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="absolute left-16 right-0 top-0 bottom-0 p-6 overflow-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-navy/40" />
              <input
                type="text"
                placeholder="Search patients..."
                className="pl-10 pr-4 py-2.5 bg-white rounded-xl text-sm text-navy placeholder:text-navy/40 border border-navy/5 focus:outline-none focus:border-mint/50 w-64"
              />
            </div>
            <div className="flex items-center gap-3">
              <button className="w-10 h-10 rounded-xl bg-white border border-navy/5 flex items-center justify-center text-navy/60 hover:text-navy transition-colors">
                <Scan className="w-5 h-5" />
              </button>
              <div className="w-10 h-10 rounded-xl bg-mint/15 flex items-center justify-center">
                <span className="text-sm font-semibold text-mint">DR</span>
              </div>
            </div>
          </div>

          {/* Stats Row - Placeholder */}
          <div ref={statsRef} className="grid grid-cols-3 gap-4 mb-6">
            {[
              { icon: Scan, label: 'Scans', color: 'mint' },
              { icon: Clock, label: 'Pending', color: 'amber' },
              { icon: TrendingUp, label: 'Accuracy', color: 'green' },
            ].map((stat, i) => (
              <div
                key={i}
                className="bg-white rounded-2xl p-5 card-shadow"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-xl bg-navy/5 flex items-center justify-center">
                    <stat.icon className="w-5 h-5 text-navy/60" />
                  </div>
                  <span className="text-sm text-navy/60">{stat.label}</span>
                </div>
                <div className="text-2xl font-bold text-navy">—</div>
              </div>
            ))}
          </div>

          {/* Bottom Row */}
          <div className="grid grid-cols-2 gap-4">
            {/* Chart Card */}
            <div
              ref={chartRef}
              className="bg-white rounded-2xl p-5 card-shadow"
            >
              <h3 className="text-sm font-semibold text-navy mb-4">Scans Overview</h3>
              <div className="h-32 flex items-center justify-center text-navy/40 text-sm">
                No data available
              </div>
            </div>

            {/* List Card */}
            <div
              ref={listRef}
              className="bg-white rounded-2xl p-5 card-shadow"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-navy">Recent Activity</h3>
                <button className="text-xs text-mint flex items-center gap-1 hover:underline">
                  View all <ChevronRight className="w-3 h-3" />
                </button>
              </div>
              <div className="text-center py-8 text-navy/40 text-sm">
                No recent activity
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
