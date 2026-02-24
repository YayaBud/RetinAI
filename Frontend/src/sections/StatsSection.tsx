import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Scan, Clock, Shield } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function StatsSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headingRef = useRef<HTMLDivElement>(null);
  const metricsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const heading = headingRef.current;
    const metrics = metricsRef.current;

    if (!section || !heading || !metrics) return;

    const ctx = gsap.context(() => {
      // Heading animation
      gsap.fromTo(
        heading,
        { y: 20, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.6,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: heading,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      // Metrics animation
      gsap.fromTo(
        metrics.children,
        { y: 24, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          stagger: 0.1,
          duration: 0.5,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: metrics,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="stats"
      className="relative w-full bg-offwhite py-20 z-[90]"
    >
      <div className="max-w-[1100px] mx-auto px-6 lg:px-12">
        {/* Heading */}
        <div ref={headingRef} className="text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-navy mb-4">
            Built for clinicians
          </h2>
          <p className="text-base text-navy/60 max-w-xl mx-auto">
            From single practices to hospital networks—OptiScan AI keeps workflow fast and data secure.
          </p>
        </div>

        {/* Metrics - Placeholder */}
        <div ref={metricsRef} className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { icon: Scan, label: 'Scans analyzed', color: 'mint' },
            { icon: Clock, label: 'Average processing', color: 'blue' },
            { icon: Shield, label: 'Uptime', color: 'green' },
          ].map((metric, i) => (
            <div
              key={i}
              className="bg-white rounded-2xl p-8 card-shadow text-center"
            >
              <div className="w-14 h-14 rounded-2xl bg-navy/5 flex items-center justify-center mx-auto mb-4">
                <metric.icon className="w-7 h-7 text-navy/60" />
              </div>
              <div className="text-3xl font-bold text-navy mb-2">—</div>
              <div className="text-sm text-navy/60">{metric.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
