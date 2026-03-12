import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { CalendarDays, Clock, ChevronLeft, ChevronRight, Plus } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function CalendarSection() {
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
      gsap.set([leftCard, rightCard], { opacity: 0, scale: 0.8 });
      gsap.set(miniCard, { opacity: 0, scale: 0.6, y: 30 });

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
          tl.to(leftCard, { scale: 1, opacity: 1 }, 0)
            .to(rightCard, { scale: 1, opacity: 1 }, 0.1)
            .to(miniCard, { scale: 1, y: 0, opacity: 1, ease: 'back.out(1.4)' }, 0.18);
        },
        once: true,
      });
    }, section);

    return () => ctx.revert();
  }, []);

  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const dates = [15, 16, 17, 18, 19, 20, 21];

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ position: 'sticky', top: 0, zIndex: 7, backgroundColor: 'rgba(8,15,23,0.55)' }}
    >
      {/* Left Media Card (Calendar UI) */}
      <div
        ref={leftCardRef}
        className="absolute rounded-[28px] overflow-hidden card-shadow"
        style={{
          left: '10vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <img
          src="/calendar_room.jpg"
          alt="Clinic reception"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-navy/30 to-transparent" />

        {/* Calendar UI Overlay */}
        <div className="absolute inset-0 flex items-center justify-center p-8">
          <div className="bg-[#0d1a28]/95 backdrop-blur-sm rounded-2xl p-6 w-full max-w-sm card-shadow">
            {/* Calendar Header */}
            <div className="flex items-center justify-between mb-6">
              <button className="w-8 h-8 rounded-lg hover:bg-navy/5 flex items-center justify-center">
                <ChevronLeft className="w-4 h-4 text-white/60" />
              </button>
              <span className="font-semibold text-white">January 2025</span>
              <button className="w-8 h-8 rounded-lg hover:bg-navy/5 flex items-center justify-center">
                <ChevronRight className="w-4 h-4 text-white/60" />
              </button>
            </div>

            {/* Days */}
            <div className="grid grid-cols-7 gap-1 mb-2">
              {days.map((day) => (
                <div key={day} className="text-center text-xs text-white/40 py-2">
                  {day}
                </div>
              ))}
            </div>

            {/* Dates */}
            <div className="grid grid-cols-7 gap-1 mb-6">
              {dates.map((date, i) => (
                <button
                  key={date}
                  className={`aspect-square rounded-lg flex items-center justify-center text-sm transition-colors ${
                    i === 3
                      ? 'bg-mint text-white font-semibold'
                      : 'hover:bg-navy/5 text-white/70'
                  }`}
                >
                  {date}
                </button>
              ))}
            </div>

            {/* Add Button */}
            <button className="w-full py-3 bg-mint/15 text-mint rounded-xl text-sm font-medium flex items-center justify-center gap-2 hover:bg-mint/25 transition-colors">
              <Plus className="w-4 h-4" />
              Add appointment
            </button>
          </div>
        </div>
      </div>

      {/* Right Content Card */}
      <div
        ref={rightCardRef}
        className="absolute bg-[#0d1a28] rounded-[28px] card-shadow border border-white/5 flex flex-col justify-center p-10"
        style={{
          left: '52vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <div className="flex items-center gap-2 mb-6">
          <CalendarDays className="w-5 h-5 text-mint" />
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
            Scheduling
          </span>
        </div>

        <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4 leading-tight">
          Schedule in a few clicks
        </h2>

        <p className="text-base text-white/50 leading-relaxed max-w-md">
          Book follow-ups, send reminders, and block time for urgent reviews—without leaving the app.
        </p>
      </div>

      {/* Mini Card */}
      <div
        ref={miniCardRef}
        className="absolute bg-[#0d1a28] rounded-[22px] card-shadow border border-white/5 p-6 flex flex-col justify-center"
        style={{
          left: '10vw',
          bottom: '10vh',
          width: '18vw',
          height: '18vh',
          minWidth: '200px',
        }}
      >
        <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
          <Clock className="w-4 h-4" />
          Next available slot
        </div>
        <div className="text-2xl font-bold text-mint">Today, 2:30 PM</div>
      </div>
    </section>
  );
}
