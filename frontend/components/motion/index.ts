export { FadeIn } from "./FadeIn";
export { StaggerContainer } from "./StaggerContainer";
export { StaggerItem } from "./StaggerItem";
export { GrowBar } from "./GrowBar";
export { useChartReveal } from "./useChartReveal";

// AnimatedNumber.tsx is intentionally not exported yet — no consumer wires
// it up in this pass, and its initial-value/reduced-motion branch needs a
// hydration-safety fix before it's adopted anywhere. Left in place for a
// future commit rather than deleted.
