import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Textarea } from "./Textarea";

describe("Textarea", () => {
  it("updates value when typing", async () => {
    const user = userEvent.setup();
    const { container } = render(<Textarea aria-label="message" />);
    const textarea = screen.getByRole("textbox");
    await user.type(textarea, "Hello");
    expect(textarea).toHaveValue("Hello");
    expect(await axe(container)).toHaveNoViolations();
  });
});
