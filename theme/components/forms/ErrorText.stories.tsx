import type { Meta, StoryObj } from "@storybook/react";
import { ErrorText } from "./ErrorText";

const meta: Meta<typeof ErrorText> = {
  component: ErrorText,
  args: {
    children: "Dieses Feld ist erforderlich.",
  },
};

export default meta;
type Story = StoryObj<typeof ErrorText>;

export const Default: Story = {};
