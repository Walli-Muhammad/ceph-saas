"use client";
import Image from "next/image";
import { motion } from "framer-motion";

function FadeUp({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay, ease: "easeOut" }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

export default function Hero() {
    return (
        <section className="relative min-h-screen flex items-center pt-20 overflow-hidden">
            {/* Background glows */}
            <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full glow-purple opacity-40 blur-3xl pointer-events-none animate-pulse-glow" />
            <div className="absolute -top-20 right-0 w-[500px] h-[500px] rounded-full glow-cyan opacity-30 blur-3xl pointer-events-none animate-pulse-glow" style={{ animationDelay: "1.5s" }} />

            <div className="max-w-7xl mx-auto px-6 w-full grid lg:grid-cols-2 gap-16 items-center py-20">
                {/* Left — text */}
                <div className="flex flex-col gap-6">
                    {/* Badge */}
                    <FadeUp delay={0.1}>
                        <span className="inline-flex items-center gap-2 glass px-3 py-1.5 rounded-full text-xs text-cyan-400 font-medium border border-cyan-500/20">
                            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                            AI-Powered Cephalometric Analysis
                        </span>
                    </FadeUp>

                    {/* Headline */}
                    <FadeUp delay={0.2}>
                        <h1 className="text-5xl lg:text-6xl font-extrabold tracking-tight leading-[1.08] text-white">
                            Precision Analysis{" "}
                            <span className="gradient-text">in Seconds.</span>
                        </h1>
                    </FadeUp>

                    {/* Subhead */}
                    <FadeUp delay={0.32}>
                        <p className="text-lg text-slate-400 max-w-xl leading-relaxed">
                            Transform orthodontic diagnosis with AI that detects landmarks automatically, measures angles instantly, and generates comprehensive clinical reports — empowering smarter treatment.
                        </p>
                    </FadeUp>

                    {/* Buttons */}
                    <FadeUp delay={0.44}>
                        <div className="flex flex-wrap gap-4 pt-2">
                            <a
                                href="#pricing"
                                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-teal-500 text-white font-semibold text-sm hover:opacity-90 active:scale-95 transition-all shadow-lg shadow-blue-500/20"
                            >
                                Get Started Free
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                </svg>
                            </a>
                            <a
                                href="https://ceph-saas-mvp.vercel.app"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass text-white font-semibold text-sm hover:border-white/20 active:scale-95 transition-all"
                            >
                                <svg className="w-4 h-4 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                Try Live Demo ↗
                            </a>
                        </div>
                    </FadeUp>

                    {/* Social proof */}
                    <FadeUp delay={0.56}>
                        <div className="flex items-center gap-3 pt-2">
                            <div className="flex -space-x-2">
                                {(["bg-cyan-500", "bg-purple-500", "bg-emerald-500", "bg-pink-500"] as const).map((c, i) => (
                                    <span key={i} className={`w-7 h-7 rounded-full ${c} border-2 border-slate-950 flex items-center justify-center text-white text-[10px] font-bold`}>
                                        {["Dr", "RK", "SA", "MP"][i]}
                                    </span>
                                ))}
                            </div>
                            <p className="text-sm text-slate-400">
                                Trusted by <span className="text-white font-semibold">500+</span> orthodontists worldwide
                            </p>
                        </div>
                    </FadeUp>
                </div>

                {/* Right — floating dashboard image */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.92, x: 40 }}
                    animate={{ opacity: 1, scale: 1, x: 0 }}
                    transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
                    className="relative flex items-center justify-center"
                >
                    {/* Glow blobs behind image */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="w-80 h-80 rounded-full glow-cyan opacity-30 blur-3xl animate-pulse-glow" />
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ transform: "translate(30px, 30px)" }}>
                        <div className="w-72 h-72 rounded-full glow-purple opacity-25 blur-3xl animate-pulse-glow" style={{ animationDelay: "1s" }} />
                    </div>

                    {/* Floating dashboard */}
                    <motion.div
                        animate={{ y: [0, -12, 0] }}
                        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                        className="relative z-10 w-full max-w-xl rounded-2xl overflow-hidden shadow-2xl shadow-black/60 border border-white/10"
                    >
                        <Image
                            src="/dashboard.png"
                            alt="CephaloAI Dashboard showing cephalometric analysis"
                            width={720}
                            height={500}
                            className="w-full h-auto"
                            priority
                        />
                    </motion.div>

                    {/* Floating stat chips */}
                    <motion.div
                        animate={{ y: [0, -6, 0] }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
                        className="absolute -left-4 top-1/4 glass rounded-xl px-4 py-3 shadow-xl border border-white/10 z-20"
                    >
                        <p className="text-xs text-slate-400">Accuracy</p>
                        <p className="text-lg font-bold text-cyan-400">99.5%</p>
                    </motion.div>

                    <motion.div
                        animate={{ y: [0, -8, 0] }}
                        transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                        className="absolute -right-4 bottom-1/4 glass rounded-xl px-4 py-3 shadow-xl border border-white/10 z-20"
                    >
                        <p className="text-xs text-slate-400">Analysis Time</p>
                        <p className="text-lg font-bold text-purple-400">2.5s</p>
                    </motion.div>
                </motion.div>
            </div >
        </section >
    );
}
