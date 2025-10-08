import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Form } from "./Form";

describe("Form", () => {
  it("renders a semantic form", async () => {
    const { container } = render(
      <Form data-testid="form" className="custom" onSubmit={(event) => event.preventDefault()}>
        <button type="submit">Submit</button>
      </Form>
    );

    const form = screen.getByTestId("form");
    expect(form).toHaveAttribute("novalidate");
    expect(form).toHaveClass("custom");
    expect(await axe(container)).toHaveNoViolations();
  });
});
