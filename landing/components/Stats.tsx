"use client";
import { useEffect, useRef } from "react";
import { motion, useInView, useMotionValue, useSpring } from "framer-motion";

interface StatProps {
    value: number;
    suffix: string;
    label: string;
    color: string;
}

function AnimatedNumber({ value, suffix, label, color }: StatProps) {
    const ref = useRef<HTMLDivElement>(null);
    const isInView = useInView(ref, { once: true, margin: "-60px" });
    const motionVal = useMotionValue(0);
    const spring = useSpring(motionVal, { stiffness: 60, damping: 18 });
    const displayRef = useRef<HTMLSpanElement>(null);

    useEffect(() => {
        if (!isInView) return;
        motionVal.set(value);
    }, [isInView, motionVal, value]);

    useEffect(
        () =>
            spring.on("change", (v) => {
                if (displayRef.current) {
                    displayRef.current.textContent =
                        v >= 1000
                            ? Math.round(v / 1000) + "K"
                            : v % 1 !== 0
                                ? v.toFixed(1)
                                : Math.round(v).toString();
                }
            }),
        [spring]
    );

    return (
        <motion.div
            ref={ref}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="flex flex-col items-center gap-2 text-center"
        >
            <div className={`text-5xl lg:text-6xl font-black tracking-tight ${color}`}>
                <span ref={displayRef}>0</span>
                <span>{suffix}</span>
            </div>
            <p className="text-slate-400 text-sm font-medium">{label}</p>
        </motion.div>
    );
}

export default function Stats() {
    return (
        <section className="py-24 px-6 relative overflow-hidden">
            {/* Big dark glass panel */}
            <div className="max-w-5xl mx-auto glass rounded-3xl border border-white/[0.07] p-16 relative overflow-hidden">
                {/* Decorative blobs inside */}
                <div className="absolute -right-20 -top-20 w-80 h-80 glow-cyan opacity-20 blur-3xl rounded-full pointer-events-none animate-pulse-glow" />
                <div className="absolute -left-20 -bottom-20 w-80 h-80 glow-purple opacity-20 blur-3xl rounded-full pointer-events-none animate-pulse-glow" style={{ animationDelay: "1.2s" }} />

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.3 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-14"
                >
                    <span className="text-xs font-semibold tracking-widest uppercase text-purple-400 mb-3 block">
                        Precision Powered by Deep Learning
                    </span>
                    <h2 className="text-3xl lg:text-4xl font-extrabold text-white tracking-tight">
                        Numbers that <span className="gradient-text">speak for themselves</span>
                    </h2>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-12 divideHorizontal">
                    <AnimatedNumber value={100000} suffix="+" label="Analyses Performed" color="text-cyan-400" />
                    <AnimatedNumber value={99.5} suffix="%" label="Landmark Accuracy" color="text-purple-400" />
                    <AnimatedNumber value={2.5} suffix="s" label="Average Analysis Time" color="text-emerald-400" />
                </div>
            </div>
        </section>
    );
}
