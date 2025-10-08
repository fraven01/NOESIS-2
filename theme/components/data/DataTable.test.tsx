import * as React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { DataTable, Row } from "./DataTable";

describe("DataTable", () => {
  const rows: Row[] = [
    { id: "1", name: "Bravo" },
    { id: "2", name: "Alpha" },
    { id: "3", name: "Charlie" },
  ];

  it("allows selecting rows", async () => {
    const user = userEvent.setup();

    const Example = () => {
      const [selected, setSelected] = React.useState<string[]>([]);
      return <DataTable rows={rows} selected={selected} onSelectedChange={setSelected} />;
    };

    const { container } = render(<Example />);

    await user.click(screen.getByRole("checkbox", { name: /select bravo/i }));
    expect(screen.getByRole("checkbox", { name: /select bravo/i })).toBeChecked();

    await user.click(screen.getByRole("checkbox", { name: /select all/i }));
    expect(screen.getAllByRole("checkbox", { checked: true })).toHaveLength(rows.length + 1);

    expect(await axe(container)).toHaveNoViolations();
  });

  it("sorts rows by name", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <DataTable rows={rows} selected={[]} onSelectedChange={() => {}} />
    );

    const header = screen.getByRole("columnheader", { name: /name/i });

    await user.click(header);
    let bodyRows = screen.getAllByRole("row").slice(1);
    expect(
      bodyRows.map((row) => within(row).getAllByRole("cell")[1].textContent)
    ).toEqual(["Alpha", "Bravo", "Charlie"]);

    await user.click(header);
    bodyRows = screen.getAllByRole("row").slice(1);
    expect(
      bodyRows.map((row) => within(row).getAllByRole("cell")[1].textContent)
    ).toEqual(["Charlie", "Bravo", "Alpha"]);

    expect(await axe(container)).toHaveNoViolations();
  });
});
