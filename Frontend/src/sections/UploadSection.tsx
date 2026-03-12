import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { CloudUpload, FileImage, FileDigit } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function UploadSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const uploadUIRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const uploadUI = uploadUIRef.current;

    if (!section || !leftCard || !rightCard || !uploadUI) return;

    const ctx = gsap.context(() => {
      gsap.set([leftCard, rightCard], { opacity: 0, rotateY: 0 });
      gsap.set(leftCard, { x: '-8vw' });
      gsap.set(rightCard, { x: '8vw' });
      gsap.set(uploadUI, { opacity: 0, y: 30, scale: 0.9 });

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
          tl.to(leftCard, { x: 0, opacity: 1 }, 0)
            .to(rightCard, { x: 0, opacity: 1 }, 0.08)
            .to(uploadUI, { y: 0, scale: 1, opacity: 1, ease: 'back.out(1.3)' }, 0.18);
        },
        once: true,
      });
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ position: 'sticky', top: 0, zIndex: 3, backgroundColor: 'rgba(8,15,23,0.55)' }}
    >
      {/* Left Content Card */}
      <div
        ref={leftCardRef}
        className="absolute bg-[#0d1a28] rounded-[28px] card-shadow border border-white/5 flex flex-col justify-center p-10"
        style={{
          left: '10vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <div className="flex items-center gap-2 mb-6">
          <CloudUpload className="w-5 h-5 text-mint" />
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
            Secure Upload
          </span>
        </div>

        <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4 leading-tight">
          Upload images securely
        </h2>

        <p className="text-base text-white/50 leading-relaxed max-w-md mb-8">
          Drag-and-drop or connect your device. We handle DICOM, JPEG, and PNG.
        </p>

        {/* File Type Indicators */}
        <div className="flex gap-4">
          {[
            { icon: FileImage, label: 'JPEG/PNG' },
            { icon: FileDigit, label: 'DICOM' },
          ].map((type) => (
            <div
              key={type.label}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-lg text-sm text-white/50"
            >
              <type.icon className="w-4 h-4" />
              {type.label}
            </div>
          ))}
        </div>
      </div>

      {/* Right Media Card */}
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
          src="/upload_room.jpg"
          alt="Modern exam room"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-navy/30 to-transparent" />

        {/* Upload UI Overlay */}
        <div
          ref={uploadUIRef}
          className="absolute inset-0 flex items-center justify-center"
        >
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-8 w-80 card-shadow">
            {/* Drop Zone */}
            <div className="border-2 border-dashed border-mint/40 rounded-xl p-6 mb-6 text-center">
              <CloudUpload className="w-10 h-10 text-mint mx-auto mb-3" />
              <p className="text-sm text-navy/70">Drop files here or click to browse</p>
            </div>

            {/* Progress */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-navy/70">Uploading 2 files...</span>
                <span className="text-mint font-semibold">65%</span>
              </div>
              <div className="h-2 bg-navy/10 rounded-full overflow-hidden">
                <div className="h-full bg-mint rounded-full animate-progress origin-left" style={{ width: '65%' }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
