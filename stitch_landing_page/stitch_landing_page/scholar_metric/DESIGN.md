# Design System Document: Academic Editorial

## 1. Overview & Creative North Star
**Creative North Star: The Modern Archive**
This design system rejects the "dashboard-as-a-utility" cliché in favor of a high-end, editorial aesthetic tailored for academia. Instead of cluttered grids, we embrace "The Modern Archive"—a philosophy where information is treated with the gravity of a scholarly journal but the fluidity of modern digital architecture. 

We break the "template" look through **intentional asymmetry** (e.g., wide margins paired with compact data clusters) and **tonal depth**. By utilizing a high-contrast typography scale and a sophisticated layering system, we transform a routine attendance portal into an authoritative, premium experience that feels both permanent and effortless.

## 2. Colors & Surface Logic
The palette is rooted in Slate and Charcoal, punctuated by a burnt ochre `tertiary` that commands attention without disrupting the scholarly calm.

### The "No-Line" Rule
To achieve a signature look, **1px solid borders are prohibited for sectioning.** Visual boundaries must be defined solely through background shifts. For example, a `surface_container_low` section should sit directly on a `surface` background. Let the change in value define the edge, not a stroke.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of fine paper. Use the surface-container tiers to create "nested" importance:
- **Base Layer:** `surface` (#f8f9fa) – The desk.
- **Section Layer:** `surface_container_low` (#f3f4f5) – The folder.
- **Primary Content Card:** `surface_container_lowest` (#ffffff) – The document.

### Signature Textures
While the system avoids "neon" or aggressive gradients, use **Subtle Tonal Transitions** for high-level interactive states. A transition from `primary` (#233343) to `primary_container` (#3a4a5a) on a hover state provides a "soul" and professional polish that a flat color swap cannot achieve.

## 3. Typography: The Editorial Voice
We use **Inter** across the board, but we treat it with editorial rigor. The hierarchy is designed to guide the eye through dense academic data.

*   **Display & Headline (The Authority):** Use `display-md` (2.75rem) for main dashboard greetings. Keep letter-spacing at -0.02em to create a "tight," custom-set feel.
*   **Titles (The Structure):** `title-lg` (1.375rem) should be used for card headings. Ensure these have significant top-margin (`spacing-8`) to allow the layout to breathe.
*   **Body (The Content):** `body-md` (0.875rem) is our workhorse. For attendance tables, use `body-sm` (0.75rem) to maintain a compact, scholarly information density.
*   **Labels (The Metadata):** Use `label-md` with `on_surface_variant` (#43474c) for secondary data.

## 4. Elevation & Depth: Tonal Layering
Traditional shadows are often a crutch for poor layout. In this system, we prioritize **Tonal Layering**.

### The Layering Principle
Depth is achieved by "stacking" the surface tiers. Place a `surface_container_lowest` card on a `surface_container_low` background. The subtle contrast (White on Off-White) creates a soft, natural lift that mimics natural light hitting paper.

### Ambient Shadows
Where floating elements (like dropdowns or modals) are required, use **Ambient Shadows**:
- **Blur:** 24px - 32px
- **Opacity:** 4% - 6%
- **Color:** Use a tinted version of `on_surface` (#191c1d) rather than pure black. This ensures the shadow feels like part of the environment.

### The "Ghost Border" Fallback
If a border is required for accessibility (e.g., in high-contrast mode), use a **Ghost Border**: the `outline_variant` token at **20% opacity**. Never use 100% opaque borders for interior elements.

## 5. Components

### Cards & Lists
*   **The Rule:** Forbid divider lines. 
*   **Execution:** Separate list items using `spacing-4` (1rem) of vertical white space or by alternating background colors between `surface_container_lowest` and `surface_container_low`.
*   **Cards:** Use `rounded-md` (0.375rem) for a crisp, professional corner.

### Buttons
*   **Primary:** `primary` background with `on_primary` text. Use `spacing-6` (1.5rem) horizontal padding for a wide, confident stance.
*   **Secondary:** `surface_container_high` background. No border. This creates a "soft" button that feels integrated into the page.

### Input Fields
*   **State Logic:** Default state uses `surface_container_highest` background. 
*   **Focus:** Transition to a `ghost border` using `primary` at 40% opacity. 
*   **Error:** Use `error` (#ba1a1a) for the label text and a 1px `error` border only in this high-alert state.

### Academic Tables (Custom Component)
Tables must feel like a modern spreadsheet. 
- **Header:** `surface_container_high` background, `label-md` text, all-caps with 0.05em tracking.
- **Rows:** No horizontal lines. Use a `surface_container_low` hover state to highlight the active row.

## 6. Do's and Don'ts

### Do
*   **Do** use asymmetrical layouts. Push a sidebar to the far left and center the main content with generous `spacing-20` gutters.
*   **Do** use `tertiary` (#562300) sparingly for "Action Required" or "Absent" states. It is a powerful color; treat it as a surgical tool.
*   **Do** prioritize vertical rhythm. Ensure all elements align to a 4px baseline to maintain the "Structured" feel.

### Don't
*   **Don't** use 100% black (#000000). Always use `on_surface` for text to maintain the "ink on paper" softness.
*   **Don't** use heavy dropshadows. If the elevation isn't clear through tonal shifts, your layout needs more white space, not more shadow.
*   **Don't** crowd the screen. Academic portals are often data-heavy; combat this by strictly adhering to the `spacing-8` (2rem) rule for container padding.