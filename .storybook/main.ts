import type { StorybookConfig } from "@storybook/react-vite";
import { mergeConfig } from "vite";
import tailwindcss from "tailwindcss";
import autoprefixer from "autoprefixer";

/**
 * Storybook configuration
 */
const config: StorybookConfig = {
  stories: ["../theme/components/**/*.stories.@(ts|tsx|mdx)"],
  addons: [
    "@storybook/addon-essentials",
    "@storybook/addon-a11y",
    "@storybook/addon-interactions",
  ],
  framework: {
    name: "@storybook/react-vite",
    options: {},
  },
  docs: {
    autodocs: "tag",
  },
  async viteFinal(viteConfig) {
    return mergeConfig(viteConfig, {
      css: {
        postcss: {
          plugins: [tailwindcss(), autoprefixer()],
        },
      },
    });
  },
};

export default config;

// TODO: Hook visual snapshot tools like Chromatic or Percy in CI.

