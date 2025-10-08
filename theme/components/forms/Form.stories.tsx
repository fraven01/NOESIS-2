import type { Meta, StoryObj } from "@storybook/react";
import { Form } from "./Form";
import { FormField } from "./FormField";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";

const meta: Meta<typeof Form> = {
  component: Form,
};

export default meta;
type Story = StoryObj<typeof Form>;

export const Default: Story = {
  render: (args) => (
    <Form {...args} onSubmit={(event) => event.preventDefault()}>
      <FormField name="email" label="E-Mail" helperText="Wir melden uns bald.">
        <Input placeholder="name@example.org" />
      </FormField>
      <Button type="submit">Absenden</Button>
    </Form>
  ),
};
