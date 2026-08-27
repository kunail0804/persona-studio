import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ScenarioEditorPage } from "./pages/ScenarioEditorPage";
import { ScenarioListPage } from "./pages/ScenarioListPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<ScenarioListPage />} />
        <Route path="/scenarios/:id" element={<ScenarioEditorPage />} />
      </Route>
    </Routes>
  );
}
