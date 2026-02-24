import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { FileCheck, AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function ResultsSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const reportLinesRef = useRef<HTMLDivElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const reportLines = reportLinesRef.current;
    const badge = badgeRef.current;

    if (!section || !leftCard || !rightCard || !reportLines || !badge) return;

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
          reportLines.children,
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, stagger: 0.04, ease: 'none' },
          0.08
        )
        .fromTo(
          badge,
          { scale: 0.85, opacity: 0 },
          { scale: 1, opacity: 1, ease: 'none' },
          0.15
        );

      // EXIT (60-100%)
      scrollTl
        .fromTo(leftCard, { x: 0, opacity: 1 }, { x: '-8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(rightCard, { x: 0, opacity: 1 }, { x: '8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(
          reportLines.children,
          { y: 0, opacity: 1 },
          { y: -10, opacity: 0.2, ease: 'power2.in' },
          0.6
        )
        .fromTo(
          badge,
          { scale: 1, opacity: 1 },
          { scale: 0.92, opacity: 0.2, ease: 'power2.in' },
          0.6
        );
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-50"
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
          <FileCheck className="w-5 h-5 text-mint" />
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
            Reports
          </span>
        </div>

        <h2 className="text-3xl lg:text-4xl font-bold text-navy mb-4 leading-tight">
          Clear, actionable reports
        </h2>

        <p className="text-base text-navy/60 leading-relaxed max-w-md">
          Get classification, confidence scores, and next-step suggestions—ready for EHR or patient sharing.
        </p>
      </div>

      {/* Right Media Card (Report UI) */}
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
          src="/report_room.jpg"
          alt="Doctor reviewing report"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-navy/40 to-navy/20" />

        {/* Report UI Overlay */}
        <div className="absolute inset-0 flex items-center justify-center p-8">
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-6 w-full max-w-sm card-shadow">
            {/* Report Header */}
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-navy">Analysis Report</h3>
              <span className="text-xs text-navy/50">#OS-2024-0847</span>
            </div>

            {/* Report Lines */}
            <div ref={reportLinesRef} className="space-y-4">
              {/* Patient Info */}
              <div className="flex items-center gap-3 p-3 bg-navy/5 rounded-xl">
                <div className="w-10 h-10 rounded-full bg-mint/20 flex items-center justify-center">
                  <span className="text-sm font-semibold text-mint">JD</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-navy">John Doe</p>
                  <p className="text-xs text-navy/50">Male, 58 years</p>
                </div>
              </div>

              {/* Finding */}
              <div className="flex items-start gap-3 p-3 bg-amber-50 rounded-xl border border-amber-100">
                <AlertCircle className="w-5 h-5 text-amber-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-700">Finding</p>
                  <p className="text-sm text-amber-600">
                    Mild diabetic retinopathy detected
                  </p>
                </div>
              </div>

              {/* Recommendation */}
              <div className="flex items-start gap-3 p-3 bg-mint/10 rounded-xl border border-mint/20">
                <CheckCircle2 className="w-5 h-5 text-mint mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-navy">Recommendation</p>
                  <p className="text-sm text-navy/70">
                    Schedule follow-up in 6 months
                  </p>
                </div>
              </div>
            </div>

            {/* Confidence Badge */}
            <div ref={badgeRef} className="mt-6 flex justify-center">
              <div className="animate-breathe flex items-center gap-2 px-5 py-2.5 bg-mint rounded-full">
                <TrendingUp className="w-4 h-4 text-navy" />
                <span className="text-sm font-semibold text-navy">Confidence 94%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
