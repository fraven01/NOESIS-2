import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Label } from "./Label";

describe("Label", () => {
  it("sets htmlFor correctly and is accessible", async () => {
    const { container } = render(<Label htmlFor="email">Email</Label>);
    const label = screen.getByText("Email");
    expect(label).toHaveAttribute("for", "email");
    expect(await axe(container)).toHaveNoViolations();
  });
});
