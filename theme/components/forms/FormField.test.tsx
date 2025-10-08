import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { FormField } from "./FormField";
import { Input } from "../ui/Input";

describe("FormField", () => {
  it("links label and helper text", async () => {
    const { container, rerender } = render(
      <FormField name="email" label="E-Mail" helperText="Pflichtfeld">
        <Input />
      </FormField>
    );

    const input = screen.getByLabelText("E-Mail");
    expect(input).toHaveAttribute("id", "email");
    expect(input).toHaveAttribute("aria-describedby", "email-helper");
    expect(await axe(container)).toHaveNoViolations();

    rerender(
      <FormField name="email" label="E-Mail" error="Ungültig">
        <Input />
      </FormField>
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Ungültig");
    expect(screen.getByLabelText("E-Mail")).toHaveAttribute("aria-invalid", "true");
    expect(await axe(container)).toHaveNoViolations();
  });
});
