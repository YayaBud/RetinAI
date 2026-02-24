import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Upload, Brain, FileText, Zap } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function PipelineSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const chipsRef = useRef<HTMLDivElement>(null);
  const miniCardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const chips = chipsRef.current;
    const miniCard = miniCardRef.current;

    if (!section || !leftCard || !rightCard || !chips || !miniCard) return;

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

      // ENTRANCE (0-40%) - Faster
      scrollTl
        .fromTo(leftCard, { x: '-50vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0)
        .fromTo(rightCard, { x: '50vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0)
        .fromTo(
          chips.children,
          { y: 30, opacity: 0 },
          { y: 0, opacity: 1, stagger: 0.05, ease: 'none' },
          0.05
        )
        .fromTo(
          miniCard,
          { y: '20vh', scale: 0.92, opacity: 0 },
          { y: 0, scale: 1, opacity: 1, ease: 'none' },
          0.1
        );

      // EXIT (60-100%) - Faster
      scrollTl
        .fromTo(leftCard, { x: 0, opacity: 1 }, { x: '-8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(rightCard, { x: 0, opacity: 1 }, { x: '8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(
          chips.children,
          { y: 0, opacity: 1 },
          { y: -15, opacity: 0.2, ease: 'power2.in' },
          0.6
        )
        .fromTo(
          miniCard,
          { y: 0, opacity: 1 },
          { y: '15vh', opacity: 0.2, ease: 'power2.in' },
          0.6
        );
    }, section);

    return () => ctx.revert();
  }, []);

  const steps = [
    { icon: Upload, label: 'Upload', active: true },
    { icon: Brain, label: 'AI Analysis', active: false },
    { icon: FileText, label: 'Report', active: false },
  ];

  return (
    <section
      ref={sectionRef}
      id="pipeline"
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-20"
    >
      {/* Left Media Card */}
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
          src="/pipeline_retina_scan.jpg"
          alt="Retina scan on monitor"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-navy/10 to-transparent" />
      </div>

      {/* Right Content Card */}
      <div
        ref={rightCardRef}
        className="absolute bg-offwhite rounded-[28px] card-shadow flex flex-col justify-center p-10"
        style={{
          left: '52vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        {/* Step Chips */}
        <div ref={chipsRef} className="flex gap-3 mb-8">
          {steps.map((step) => (
            <div
              key={step.label}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-all ${
                step.active
                  ? 'bg-mint text-navy'
                  : 'bg-navy/5 text-navy/60 hover:bg-navy/10'
              }`}
            >
              <step.icon className="w-4 h-4" />
              {step.label}
            </div>
          ))}
        </div>

        {/* Title */}
        <h2 className="text-3xl lg:text-4xl font-bold text-navy mb-4 leading-tight">
          From image to insight—in seconds
        </h2>

        {/* Body */}
        <p className="text-base text-navy/60 leading-relaxed max-w-md">
          Our AI pipeline detects anomalies, generates attention maps, and classifies risk across four major eye conditions.
        </p>
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
          <Zap className="w-4 h-4" />
          Typical scan time
        </div>
        <div className="text-4xl font-bold text-mint">&lt; 4 sec</div>
      </div>
    </section>
  );
}
