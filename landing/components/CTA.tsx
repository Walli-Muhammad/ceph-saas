"use client";
import { motion } from "framer-motion";

export default function CTA() {
    return (
        <section className="py-28 px-6">
            <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.7, ease: "easeOut" }}
                className="max-w-4xl mx-auto rounded-3xl overflow-hidden relative"
            >
                {/* Gradient background */}
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-600/30 via-purple-600/30 to-slate-900 z-0" />
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(168,85,247,0.3),transparent_60%)] z-0" />
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,rgba(34,211,238,0.2),transparent_60%)] z-0" />
                <div className="absolute inset-0 border border-white/10 rounded-3xl z-0" />

                {/* Content */}
                <div className="relative z-10 text-center py-20 px-10">
                    <motion.h2
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="text-4xl lg:text-5xl font-extrabold text-white tracking-tight mb-4"
                    >
                        Ready to Transform{" "}
                        <span className="gradient-text">Your Practice?</span>
                    </motion.h2>
                    <motion.p
                        initial={{ opacity: 0, y: 16 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="text-slate-300 text-lg mb-10 max-w-xl mx-auto"
                    >
                        Join 500+ orthodontists saving hours every day with AI-powered cephalometric precision. Start your free 14-day trial now.
                    </motion.p>
                    <motion.a
                        href="#"
                        initial={{ opacity: 0, y: 16 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        whileHover={{ scale: 1.04 }}
                        whileTap={{ scale: 0.97 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                        className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-white text-slate-900 font-bold text-sm shadow-xl hover:bg-slate-100 transition-colors"
                    >
                        Get Started Free
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                        </svg>
                    </motion.a>
                </div>
            </motion.div>
        </section>
    );
}
