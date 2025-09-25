import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Input } from "./Input";

describe("Input", () => {
  it("updates value when typing", async () => {
    const user = userEvent.setup();
    const { container } = render(<Input aria-label="name" />);
    const input = screen.getByRole("textbox");
    await user.type(input, "Hello");
    expect(input).toHaveValue("Hello");
    expect(await axe(container)).toHaveNoViolations();
  });
});
