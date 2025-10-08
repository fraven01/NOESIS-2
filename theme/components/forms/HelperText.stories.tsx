import type { Meta, StoryObj } from "@storybook/react";
import { HelperText } from "./HelperText";

const meta: Meta<typeof HelperText> = {
  component: HelperText,
  args: {
    children: "Wir nutzen deine Daten nur für Rückfragen.",
  },
};

export default meta;
type Story = StoryObj<typeof HelperText>;

export const Default: Story = {};
