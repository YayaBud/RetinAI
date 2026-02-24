import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Users, Search, Filter, TrendingUp, UserPlus } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function PatientsSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const miniCardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const miniCard = miniCardRef.current;

    if (!section || !leftCard || !rightCard || !miniCard) return;

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
        .fromTo(leftCard, { x: '-50vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0)
        .fromTo(rightCard, { x: '50vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0)
        .fromTo(
          miniCard,
          { y: '20vh', scale: 0.92, opacity: 0 },
          { y: 0, scale: 1, opacity: 1, ease: 'none' },
          0.1
        );

      // EXIT (60-100%)
      scrollTl
        .fromTo(leftCard, { x: 0, opacity: 1 }, { x: '-8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(rightCard, { x: 0, opacity: 1 }, { x: '8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(
          miniCard,
          { y: 0, opacity: 1 },
          { y: '15vh', opacity: 0.2, ease: 'power2.in' },
          0.6
        );
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="patients"
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-[80]"
    >
      {/* Left Content Card */}
      <div
        ref={leftCardRef}
        className="absolute bg-offwhite rounded-[28px] card-shadow flex flex-col justify-center p-10"
        style={{
          left: '10vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <div className="flex items-center gap-2 mb-6">
          <Users className="w-5 h-5 text-mint" />
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
            Patient Management
          </span>
        </div>

        <h2 className="text-3xl lg:text-4xl font-bold text-navy mb-4 leading-tight">
          Manage patients at a glance
        </h2>

        <p className="text-base text-navy/60 leading-relaxed max-w-md">
          Filter by risk, review history, and message patients—all from one clean list.
        </p>
      </div>

      {/* Right Media Card (Patient Table UI) */}
      <div
        ref={rightCardRef}
        className="absolute rounded-[28px] overflow-hidden card-shadow"
        style={{
          left: '52vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <img
          src="/patients_room.jpg"
          alt="Clinic corridor"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-navy/40 to-navy/20" />

        {/* Patient Table UI Overlay */}
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-5 w-full card-shadow">
            {/* Table Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="relative flex-1 max-w-xs">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-navy/40" />
                <input
                  type="text"
                  placeholder="Search patients..."
                  className="w-full pl-10 pr-4 py-2 bg-navy/5 rounded-lg text-sm text-navy placeholder:text-navy/40 focus:outline-none focus:ring-2 focus:ring-mint/30"
                />
              </div>
              <div className="flex gap-2">
                <button className="flex items-center gap-2 px-3 py-2 bg-navy/5 rounded-lg text-sm text-navy/70 hover:bg-navy/10 transition-colors">
                  <Filter className="w-4 h-4" />
                  Filter
                </button>
                <button className="flex items-center gap-2 px-3 py-2 bg-mint rounded-lg text-sm text-navy font-medium hover:bg-mint/90 transition-colors">
                  <UserPlus className="w-4 h-4" />
                  Add
                </button>
              </div>
            </div>

            {/* Empty State */}
            <div className="text-center py-12">
              <div className="w-16 h-16 rounded-2xl bg-navy/5 flex items-center justify-center mx-auto mb-4">
                <Users className="w-8 h-8 text-navy/30" />
              </div>
              <p className="text-navy/50">No patients added yet</p>
            </div>
          </div>
        </div>
      </div>

      {/* Mini Card */}
      <div
        ref={miniCardRef}
        className="absolute bg-white rounded-[22px] card-shadow p-6 flex flex-col justify-center"
        style={{
          right: '8vw',
          bottom: '10vh',
          width: '18vw',
          height: '18vh',
          minWidth: '200px',
        }}
      >
        <div className="flex items-center gap-2 text-navy/50 text-sm mb-2">
          <TrendingUp className="w-4 h-4" />
          New this week
        </div>
        <div className="text-4xl font-bold text-mint">—</div>
      </div>
    </section>
  );
}
