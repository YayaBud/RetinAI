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
          uploadUI,
          { y: 30, scale: 0.96, opacity: 0 },
          { y: 0, scale: 1, opacity: 1, ease: 'none' },
          0.05
        );

      // EXIT (60-100%)
      scrollTl
        .fromTo(leftCard, { x: 0, opacity: 1 }, { x: '-8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(rightCard, { x: 0, opacity: 1 }, { x: '8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(
          uploadUI,
          { y: 0, opacity: 1 },
          { y: 15, opacity: 0.25, ease: 'power2.in' },
          0.6
        );
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-30"
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
          <CloudUpload className="w-5 h-5 text-mint" />
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
            Secure Upload
          </span>
        </div>

        <h2 className="text-3xl lg:text-4xl font-bold text-navy mb-4 leading-tight">
          Upload images securely
        </h2>

        <p className="text-base text-navy/60 leading-relaxed max-w-md mb-8">
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
              className="flex items-center gap-2 px-4 py-2 bg-navy/5 rounded-lg text-sm text-navy/70"
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
