import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { ErrorState } from "./ErrorState";

describe("ErrorState", () => {
  it("renders content, has alert role and handles CTA", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    const { container } = render(
      <ErrorState title="Error" text="Something went wrong" cta={{ label: "Retry", onClick }} />
    );
    expect(screen.getByRole("heading", { name: "Error" })).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "Retry" });
    await user.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});
