import React from "react";
import type { Preview } from "@storybook/react";
import "../theme/static_src/input.css";

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: "^on[A-Z].*" },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
    a11y: { element: "#root" },
  },
  globalTypes: {
    theme: {
      name: "Theme",
      defaultValue: "light",
      toolbar: {
        icon: "circlehollow",
        items: [
          { value: "light", title: "Light" },
          { value: "dark", title: "Dark" },
        ],
        showName: true,
      },
    },
    direction: {
      name: "Direction",
      defaultValue: "ltr",
      toolbar: {
        icon: "globe",
        items: [
          { value: "ltr", title: "LTR" },
          { value: "rtl", title: "RTL" },
        ],
        showName: true,
      },
    },
  },
  decorators: [
    (Story, context) => {
      const { theme, direction } = context.globals;
      const html = document.documentElement;
      html.classList.toggle("dark", theme === "dark");
      html.dir = direction;
      return <Story />;
    },
  ],
};

export default preview;

