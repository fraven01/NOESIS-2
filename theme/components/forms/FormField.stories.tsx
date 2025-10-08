import type { Meta, StoryObj } from "@storybook/react";
import { FormField } from "./FormField";
import { Input } from "../ui/Input";

const meta: Meta<typeof FormField> = {
  component: FormField,
  args: {
    name: "email",
    label: "E-Mail",
    helperText: "Wir melden uns bald.",
  },
};

export default meta;
type Story = StoryObj<typeof FormField>;

export const Default: Story = {
  render: (args) => (
    <FormField {...args}>
      <Input placeholder="name@example.org" />
    </FormField>
  ),
};

export const WithError: Story = {
  render: (args) => (
    <FormField {...args} error="Bitte überprüfe deine Eingabe">
      <Input placeholder="name@example.org" />
    </FormField>
  ),
};
